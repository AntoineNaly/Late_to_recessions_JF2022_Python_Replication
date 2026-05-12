# -*- coding: utf-8 -*-

"""
generate_xt_sv.py
-----------------
Kalman filter and Gibbs draw for the common growth factor z_t in the
state-space model defined by equations (IA.13)–(IA.14):

    Measurement:  y_t   = A_t (H0 + H1 x_t + v_t),   Var(v_t) = RR      (IA.13)
    Transition:   x_{t+1} = F0_t + F1 x_t + ω_{t+1},
                            Var(ω_{t+1}) = Q                               (IA.14)

The full state vector x_t has dimension nDim = 6 + N_m + N_q*5, holding the
common growth factor z_t and its 5 lags together with idiosyncratic components
e_{i,t} for all monthly and quarterly series.

Conditioning on the current parameter draws and the imputed regime path S_t,
the Kalman filter computes the filtered distribution p(x_t | y_{1:t}); a draw
from this distribution is the Gibbs step for the latent state path.

State maintained
----------------
The transition matrix F1 has non-zeros only in rows 0–5 (the z-factor lag
companion block):
    F1[0, 0]     = φ_z          AR(1) persistence of z_t (eq. 14)
    F1[1:6, 0:5] = eye(5)       lag companion: z_{t-k+1} = z_{t-k}
    F1[6:, :]    = 0            idiosyncratic AR is absorbed into H1 via
                                 pre-whitening (eq. IA.15); it is not in F1

Because F1 rows 6+ are zero, at every prediction step
    Phat = F1 @ Pt @ F1.T + Q
is non-zero only in the 6×6 z-factor block.  The idiosyncratic block resets
to diag(SIG2_i) = Q_ee at every step.  The Kalman filter therefore maintains
only the z-factor state At_z (6,) and covariance Pt_zz (6×6).

The innovation covariance is
    Ft = H1_z_M @ Phat_zz @ H1_z_M.T + diag(Q_e_sel)
where H1_z_M is the NaN-masked z-loading block (eq. IA.13) and Q_e_sel[i]
is the idiosyncratic variance contribution for observable i:
    monthly obs i:     Q_e_sel[i] = SIG2_m[i]          (from H1_e identity block)
    quarterly obs j:   Q_e_sel[j] = SIG2_q[j] / 9      (from tau_aux[0]² = 1/9, eq. 17)

Gibbs draw
----------
At each step t, z_t is drawn from its marginal filtered distribution:
    z_t         ~ N(At_z[0],  Pt_zz[0,0])
    z_{t-1} lag ~ N(At_z[1],  Pt_zz[1,1])
These scalar draws are used to reconstruct the z_t path returned to the caller.

Time-aggregation of quarterly observables is handled via the time-varying
selection matrix A_t (equation IA.13): at quarter-end months A_t = A_last
which applies Mariano–Murasawa weights (eq. 17); at other months A_t = A_NotLast
selects only the monthly rows.

Warm-start
----------
Pt_zz is passed between Gibbs iterations as Pt_prev.  On the first call
(Pt_prev=None), the unconditional covariance is obtained from the 6×6
discrete Lyapunov equation P = F1_zz P F1_zz' + Q_zz.

Outputs
-------
loglh    : float         log-likelihood summed over t (used in MH step)
z_t      : (Tstar+1, 3) columns = [draw, filtered, predicted] of z_t path
Pt_final : (6, 6)        final filtered z-factor covariance — pass back as Pt_prev
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

try:
    from .get_coefficients_sv import get_coefficients_sv
except ImportError:
    from get_coefficients_sv import get_coefficients_sv

_LOG2PI = np.log(2.0 * np.pi)

# ---------------------------------------------------------------------------
# Numba JIT kernel
# ---------------------------------------------------------------------------
_NUMBA_AVAILABLE = False
try:
    from numba import njit as _njit

    @_njit(cache=True, fastmath=True)
    def _chol_lower_nb(A):
        """Lower Cholesky factor of positive-definite matrix A."""
        n = A.shape[0]; L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                s = A[i, j]
                for k in range(j): s -= L[i, k] * L[j, k]
                L[i, j] = np.sqrt(max(s, 1e-15)) if i == j else s / L[j, j]
        return L

    @_njit(cache=True, fastmath=True)
    def _chol_solve_nb(A, b):
        """Solve A x = b (A PD) via Cholesky; return (x, log|A|). b is (n, m)."""
        L = _chol_lower_nb(A)
        n = L.shape[0]; m = b.shape[1]
        y = np.zeros((n, m)); x = np.zeros((n, m)); logdet = 0.0
        for i in range(n): logdet += 2.0 * np.log(L[i, i])
        for j in range(m):
            for i in range(n):
                s = b[i, j]
                for k in range(i): s -= L[i, k] * y[k, j]
                y[i, j] = s / L[i, i]
            for i in range(n - 1, -1, -1):
                s = y[i, j]
                for k in range(i + 1, n): s -= L[k, i] * x[k, j]
                x[i, j] = s / L[i, i]
        return x, logdet

    @_njit(cache=True, fastmath=True)
    def _kalman_loop_nb_opt(
        phi_cc,         # φ_z: AR(1) persistence of z_t (eq. 14)
        Q_z,            # σ²_{z,0}: common-factor noise variance (eq. 14), always 1.0
        F0_z,           # (Tstar,): z-factor intercept μ_z(S_{t+1}) - φ_z μ_z(S_t) (eq. 14)
        H1_z_AL,        # (N_m+N_q, 6): z-loadings of A_last @ H1   (eq. IA.13)
        H1_z_NL,        # (N_m,     6): z-loadings of A_NotLast @ H1 (eq. IA.13)
        Q_e_sel_AL,     # (N_m+N_q,): idiosyncratic Q per observable, quarter-end months
        Q_e_sel_NL,     # (N_m,):     idiosyncratic Q per observable, non-quarter-end
        is_qend,        # (Tstar,) bool: True at quarter-end months
        nan_mask_m,     # (Tstar, N_m) bool: True = valid monthly observation
        nan_mask_q,     # (Tstar, N_q) bool: True = valid quarterly observation
        Ym,             # (Tstar, N_m): pre-whitened monthly data  (eq. IA.15)
        Yq,             # (Tstar, N_q): pre-whitened quarterly data
        At_z0,          # (6,): initial z-factor state mean
        Pt_zz0,         # (6, 6): initial z-factor covariance
        randn_draw,     # (Tstar, 2): pre-drawn N(0,1) for z_t and z_{t-1} lag draws
        jitter,         # small constant added to diagonal before Cholesky
    ):
        """
        Forward Kalman filter for the z-factor state (eqs. IA.13–IA.14).

        Maintains the 6-dim z-factor state At_z and its 6×6 covariance Pt_zz.
        At each step t, draws z_t and z_{t-1} from their marginal filtered
        distributions N(At_z[0], Pt_zz[0,0]) and N(At_z[1], Pt_zz[1,1]).

        Returns
        -------
        z_draw_0 : (Tstar,)  Gibbs draw of z_t    ~ N(At_z[0], Pt_zz[0,0])
        z_draw_1 : (Tstar,)  Gibbs draw of z_{t-1} ~ N(At_z[1], Pt_zz[1,1])
        z_filt_0 : (Tstar,)  filtered mean At_z[0]
        z_filt_1 : (Tstar,)  filtered mean At_z[1]
        z_pred_0 : (Tstar,)  predicted mean α̂_{z,t|t-1}[0]
        loglh    : float      accumulated log-likelihood Σ_t log p(y_t | F_{t-1})
        Pt_zz    : (6, 6)     final filtered z-factor covariance (warm-start token)
        """
        Tstar = Ym.shape[0]
        Nm    = Ym.shape[1]
        Nq    = Yq.shape[1]

        z_draw_0 = np.empty(Tstar)
        z_draw_1 = np.empty(Tstar)
        z_filt_0 = np.empty(Tstar)
        z_filt_1 = np.empty(Tstar)
        z_pred_0 = np.empty(Tstar)
        loglh    = 0.0

        At_z  = At_z0.copy()
        Pt_zz = Pt_zz0.copy()
        JITTER = jitter

        for t in range(Tstar):

            # ----------------------------------------------------------
            # Prediction step (eq. IA.14):
            #   α̂_{z,t|t-1} = F1_zz @ At_z + F0_z[t]
            #   Phat_zz     = F1_zz @ Pt_zz @ F1_zz.T + Q_zz
            #
            # F1_zz companion form (z-factor block of F1):
            #   row 0: [φ_z, 0, 0, 0, 0, 0]  — AR(1) for z_t (eq. 14)
            #   row k: unit vector e_{k-1}    — lag shift z_{t-k+1} = z_{t-k}
            # ----------------------------------------------------------
            alphahat_z0 = phi_cc * At_z[0] + F0_z[t]   # z_t predicted mean: φ_z z_{t-1} + F0_t
            alphahat_z1 = At_z[0]                        # z_{t-1} = z_t from previous step
            alphahat_z2 = At_z[1]                        # z_{t-2} = z_{t-1} from previous step
            alphahat_z3 = At_z[2]
            alphahat_z4 = At_z[3]
            alphahat_z5 = At_z[4]

            # F1_zz @ Pt_zz rows (companion form applied row-by-row):
            #   row 0: φ_z * Pt_zz[0, :]
            #   row k: Pt_zz[k-1, :]  for k = 1..5
            F1P = np.empty((6, 6))
            for c in range(6):
                F1P[0, c] = phi_cc * Pt_zz[0, c]
                F1P[1, c] = Pt_zz[0, c]
                F1P[2, c] = Pt_zz[1, c]
                F1P[3, c] = Pt_zz[2, c]
                F1P[4, c] = Pt_zz[3, c]
                F1P[5, c] = Pt_zz[4, c]

            # (F1_zz @ Pt_zz) @ F1_zz.T: F1_zz.T columns are:
            #   col 0: [φ_z; 0; 0; 0; 0; 0]  col 1: [1; 0; 0; 0; 0; 0]
            #   col k: e_{k-1} for k=1..5
            Phat_zz = np.empty((6, 6))
            for i in range(6):
                Phat_zz[i, 0] = phi_cc * F1P[i, 0]
                Phat_zz[i, 1] = F1P[i, 0]
                Phat_zz[i, 2] = F1P[i, 1]
                Phat_zz[i, 3] = F1P[i, 2]
                Phat_zz[i, 4] = F1P[i, 3]
                Phat_zz[i, 5] = F1P[i, 4]

            # Add Q_zz: σ²_{z,0} enters only the [0,0] element of Q (eq. 14)
            # h_cc = 0 always (equal volatility across regimes), so Q_zz[0,0] = σ²_{z,0}
            Phat_zz[0, 0] += Q_z

            # Symmetrise for numerical stability
            for i in range(6):
                for j in range(i + 1, 6):
                    avg = 0.5 * (Phat_zz[i, j] + Phat_zz[j, i])
                    Phat_zz[i, j] = avg; Phat_zz[j, i] = avg

            # ----------------------------------------------------------
            # Select measurement branch (eq. IA.13):
            #   quarter-end:     A_t = A_last    → monthly + quarterly obs
            #   non-quarter-end: A_t = A_NotLast → monthly obs only
            # ----------------------------------------------------------
            if is_qend[t]:
                H1_z_A    = H1_z_AL
                Q_e_sel_A = Q_e_sel_AL
            else:
                H1_z_A    = H1_z_NL
                Q_e_sel_A = Q_e_sel_NL

            # Count valid (non-NaN) observations at this time step
            nv = 0
            for i in range(Nm):
                if nan_mask_m[t, i]: nv += 1
            if is_qend[t]:
                for i in range(Nq):
                    if nan_mask_q[t, i]: nv += 1

            # Build NaN-masked z-loading matrix H1_z_M and observation vector y_m
            H1_z_M = np.empty((nv, 6))
            y_m    = np.empty(nv)
            Q_e_M  = np.empty(nv)     # idiosyncratic Q for each valid observable

            ri = 0
            for i in range(Nm):
                if nan_mask_m[t, i]:
                    for c in range(6): H1_z_M[ri, c] = H1_z_A[i, c]
                    y_m[ri]   = Ym[t, i]
                    Q_e_M[ri] = Q_e_sel_A[i]
                    ri += 1
            if is_qend[t]:
                for i in range(Nq):
                    if nan_mask_q[t, i]:
                        for c in range(6): H1_z_M[ri, c] = H1_z_A[Nm + i, c]
                        y_m[ri]   = Yq[t, i]
                        Q_e_M[ri] = Q_e_sel_A[Nm + i]
                        ri += 1

            # ----------------------------------------------------------
            # Innovation (eq. IA.13):
            #   ν_t = y_t - H1_z_M @ α̂_{z,t|t-1}
            #
            # H0 = 0 (z_t is demeaned via F0_t).
            # The idiosyncratic predicted mean α̂_{e,t|t-1} = 0 always because
            # F1 rows 6+ are zero and F0 rows 6+ are zero.
            # ----------------------------------------------------------
            nut = np.empty(nv)
            az0 = alphahat_z0; az1 = alphahat_z1; az2 = alphahat_z2
            az3 = alphahat_z3; az4 = alphahat_z4; az5 = alphahat_z5
            for i in range(nv):
                h = H1_z_M[i]
                nut[i] = (y_m[i]
                          - h[0]*az0 - h[1]*az1 - h[2]*az2
                          - h[3]*az3 - h[4]*az4 - h[5]*az5)

            # ----------------------------------------------------------
            # Innovation covariance (eq. IA.13):
            #   Ft = H1_z_M @ Phat_zz @ H1_z_M.T + diag(Q_e_M)
            #
            # The second term is the idiosyncratic contribution
            # H1_e_M @ Phat_ee @ H1_e_M.T, which is diagonal because
            # H1_e_M is identity-like (each observable maps to one state)
            # and Phat_ee = diag(SIG2_i) at every prediction step.
            # ----------------------------------------------------------
            HP = np.zeros((nv, 6))     # H1_z_M @ Phat_zz
            for i in range(nv):
                for k in range(6):
                    s = 0.0
                    for j in range(6): s += H1_z_M[i, j] * Phat_zz[j, k]
                    HP[i, k] = s

            Ft = np.zeros((nv, nv))    # innovation covariance matrix
            for i in range(nv):
                for j in range(nv):
                    s = 0.0
                    for k in range(6): s += HP[i, k] * H1_z_M[j, k]
                    Ft[i, j] = s
                Ft[i, i] += Q_e_M[i]  # add idiosyncratic diagonal contribution

            # Symmetrise
            for i in range(nv):
                for j in range(i + 1, nv):
                    avg = 0.5 * (Ft[i, j] + Ft[j, i])
                    Ft[i, j] = avg; Ft[j, i] = avg

            # Cholesky-based solve: Ft^{-1} ν_t and log|Ft|
            nut_2d            = nut.reshape(nv, 1)
            invFt_nut, logdet = _chol_solve_nb(Ft, nut_2d)

            # Log-likelihood contribution: log N(ν_t; 0, Ft) (used in MH step)
            dot_val = 0.0
            for _ii in range(nv): dot_val += nut[_ii] * invFt_nut[_ii, 0]
            loglh += (-0.5 * nv * np.log(2.0 * np.pi)
                      - 0.5 * logdet
                      - 0.5 * dot_val)

            # ----------------------------------------------------------
            # Update step (eq. IA.14):
            #   Kalman gain: K_z = Phat_zz @ H1_z_M.T @ Ft^{-1}
            #   State update:  At_z += K_z @ ν_t
            #   Cov update:    Pt_zz = Phat_zz - K_z @ H1_z_M @ Phat_zz
            # ----------------------------------------------------------
            Ph1_z = np.zeros((6, nv))   # Phat_zz @ H1_z_M.T  (numerator of Kalman gain)
            for i in range(6):
                for j in range(nv):
                    s = 0.0
                    for k in range(6): s += Phat_zz[i, k] * H1_z_M[j, k]
                    Ph1_z[i, j] = s

            # α̂_{z,t|t} = α̂_{z,t|t-1} + K_z ν_t
            At_z[0] = az0; At_z[1] = az1; At_z[2] = az2
            At_z[3] = az3; At_z[4] = az4; At_z[5] = az5
            for i in range(6):
                s = 0.0
                for j in range(nv): s += Ph1_z[i, j] * invFt_nut[j, 0]
                At_z[i] += s

            # P_{z,t|t} = Phat_zz - K_z H1_z_M Phat_zz
            invFt_HP, _ = _chol_solve_nb(Ft, np.ascontiguousarray(HP))
            for i in range(6):
                for j in range(6):
                    s = 0.0
                    for k in range(nv): s += Ph1_z[i, k] * invFt_HP[k, j]
                    Pt_zz[i, j] = Phat_zz[i, j] - s

            # Symmetrise for numerical stability
            for i in range(6):
                for j in range(i + 1, 6):
                    avg = 0.5 * (Pt_zz[i, j] + Pt_zz[j, i])
                    Pt_zz[i, j] = avg; Pt_zz[j, i] = avg

            # Store filtered and predicted z_t means for output reconstruction
            z_filt_0[t] = At_z[0]    # α̂_{z,t|t}[0]     = filtered z_t
            z_filt_1[t] = At_z[1]    # α̂_{z,t|t}[1]     = filtered z_{t-1}
            z_pred_0[t] = az0         # α̂_{z,t|t-1}[0]   = predicted z_t

            # ----------------------------------------------------------
            # Gibbs draw: sample z_t from its marginal filtered distribution.
            # From the filtered distribution p(x_t | y_{1:t}) = N(At, Pt),
            # the marginal for each component k is N(At_z[k], Pt_zz[k,k]).
            # A small jitter is added to Pt_zz[k,k] to guard against
            # near-zero variance from floating-point accumulation.
            # ----------------------------------------------------------
            z_draw_0[t] = At_z[0] + np.sqrt(Pt_zz[0, 0] + JITTER) * randn_draw[t, 0]
            z_draw_1[t] = At_z[1] + np.sqrt(Pt_zz[1, 1] + JITTER) * randn_draw[t, 1]

        return z_draw_0, z_draw_1, z_filt_0, z_filt_1, z_pred_0, loglh, Pt_zz

    # Trigger JIT compilation on a tiny problem so the first real call is instant.
    def _warmup_kalman():
        _T  = 5; _Nm = 3; _Nq = 1
        _H1_z_AL = np.random.randn(_Nm + _Nq, 6)
        _H1_z_NL = np.random.randn(_Nm, 6)
        _Q_e_AL  = np.ones(_Nm + _Nq) * 0.3
        _Q_e_NL  = np.ones(_Nm) * 0.3
        _qend    = np.array([False]*3 + [True] + [False], dtype=np.bool_)
        _mask_m  = np.ones((_T, _Nm), dtype=np.bool_)
        _mask_q  = np.ones((_T, _Nq), dtype=np.bool_)
        _Ym      = np.random.randn(_T, _Nm)
        _Yq      = np.random.randn(_T, _Nq)
        _Az0     = np.zeros(6)
        _Pz0     = np.eye(6) * 0.5
        _rnd     = np.random.randn(_T, 2)
        _kalman_loop_nb_opt(
            0.9, 1.0, np.zeros(_T),
            _H1_z_AL, _H1_z_NL, _Q_e_AL, _Q_e_NL,
            _qend, _mask_m, _mask_q, _Ym, _Yq,
            _Az0, _Pz0, _rnd, 1e-9,
        )

    _warmup_kalman()
    _NUMBA_AVAILABLE = True
    print("[generate_xt_sv] Numba kernel compiled and cached.")

except Exception as _e:
    print(f"[generate_xt_sv] Numba unavailable ({_e}); using NumPy fallback.")


# ---------------------------------------------------------------------------
# NumPy fallback (same logic as the Numba kernel, using SciPy Cholesky)
# ---------------------------------------------------------------------------
def _kalman_loop_numpy_opt(
    phi_cc, Q_z, F0_z,
    H1_z_AL, H1_z_NL, Q_e_sel_AL, Q_e_sel_NL,
    is_qend, nan_mask_m, nan_mask_q,
    Ym, Yq, At_z, Pt_zz, randn_draw, jitter,
):
    from scipy.linalg import cho_factor, cho_solve

    Tstar = Ym.shape[0]; Nm = Ym.shape[1]; Nq = Yq.shape[1]
    z_draw_0 = np.empty(Tstar); z_draw_1 = np.empty(Tstar)
    z_filt_0 = np.empty(Tstar); z_filt_1 = np.empty(Tstar)
    z_pred_0 = np.empty(Tstar); loglh = 0.0

    # F1_zz companion matrix (z-factor block of F1, eq. 14)
    F1_zz = np.zeros((6, 6)); F1_zz[0, 0] = phi_cc; F1_zz[1:6, 0:5] = np.eye(5)

    At_z  = At_z.copy()
    Pt_zz = Pt_zz.copy()

    for t in range(Tstar):

        # Prediction step (eq. IA.14)
        alphahat_z = F1_zz @ At_z; alphahat_z[0] += F0_z[t]   # α̂_{z,t|t-1} = F1_zz At_z + F0_z
        F1P        = F1_zz @ Pt_zz
        Phat_zz    = F1P @ F1_zz.T                              # F1_zz Pt_zz F1_zz.T
        Phat_zz[0, 0] += Q_z                                    # add Q_zz[0,0] = σ²_{z,0}
        Phat_zz    = 0.5 * (Phat_zz + Phat_zz.T)               # symmetrise

        # Select measurement branch (eq. IA.13): A_last at quarter-end, A_NotLast otherwise
        if is_qend[t]:
            H1_z_A = H1_z_AL; Q_e_A = Q_e_sel_AL
        else:
            H1_z_A = H1_z_NL; Q_e_A = Q_e_sel_NL

        # NaN mask for this time step
        m_m = nan_mask_m[t]; m_q = nan_mask_q[t]
        mask_full = np.concatenate([m_m, m_q]) if is_qend[t] else m_m
        H1_z_M = H1_z_A[mask_full, :]           # NaN-masked z-loadings
        Q_e_M  = Q_e_A[mask_full]               # NaN-masked idiosyncratic Q
        y_obs  = (np.concatenate([Ym[t], Yq[t]]) if is_qend[t] else Ym[t])[mask_full]
        nv     = int(mask_full.sum())

        # Innovation ν_t = y_t - H1_z_M @ α̂_{z,t|t-1}  (H0=0; idiosyncratic mean=0)
        nut = y_obs - H1_z_M @ alphahat_z

        # Innovation covariance Ft = H1_z_M @ Phat_zz @ H1_z_M.T + diag(Q_e_M)
        HP  = H1_z_M @ Phat_zz
        Ft  = HP @ H1_z_M.T + np.diag(Q_e_M)
        Ft  = 0.5 * (Ft + Ft.T)

        # Cholesky-based solve for log-likelihood and Kalman gain
        Ft_c    = cho_factor(Ft)
        logdet  = 2.0 * np.sum(np.log(np.abs(np.diag(Ft_c[0]))))
        invFt_nut = cho_solve(Ft_c, nut.reshape(-1, 1))

        # Log-likelihood contribution: log N(ν_t; 0, Ft)
        loglh += (-0.5 * nv * _LOG2PI - 0.5 * logdet
                  - 0.5 * float((nut @ invFt_nut).item()))

        # Update step: Kalman gain K_z = Phat_zz @ H1_z_M.T @ Ft^{-1}
        Ph1_z = Phat_zz @ H1_z_M.T                        # P_{t|t-1} H1' (Kalman gain numerator)
        At_z  = alphahat_z + Ph1_z @ invFt_nut.ravel()    # α̂_{z,t|t} = α̂_{z,t|t-1} + K_z ν_t
        Pt_zz = Phat_zz - Ph1_z @ cho_solve(Ft_c, HP)     # P_{z,t|t} = Phat_zz - K_z H1_z_M Phat_zz
        Pt_zz = 0.5 * (Pt_zz + Pt_zz.T)

        z_filt_0[t] = At_z[0]; z_filt_1[t] = At_z[1]; z_pred_0[t] = alphahat_z[0]

        # Gibbs draw from marginal filtered distributions N(At_z[k], Pt_zz[k,k])
        z_draw_0[t] = At_z[0] + np.sqrt(Pt_zz[0, 0] + jitter) * randn_draw[t, 0]
        z_draw_1[t] = At_z[1] + np.sqrt(Pt_zz[1, 1] + jitter) * randn_draw[t, 1]

    return z_draw_0, z_draw_1, z_filt_0, z_filt_1, z_pred_0, loglh, Pt_zz


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_xt_sv(
    yy_monthly,
    yy_quarterly,
    s_t,
    param_macro_MH,
    param_macro_gibbs,
    indexQuarter,
    rng,
    Pt_prev=None,
):
    """
    Parameters
    ----------
    yy_monthly, yy_quarterly : (T, N_m/N_q) ndarray
    s_t                      : (T,) regime path (0=expansion, 1=recession)
    param_macro_MH           : dict — φ_z, μ_z, σ²_{z,0}, h_z, p, q
    param_macro_gibbs        : dict — γ_i, ψ_i, σ²_{e,i} for all series
    indexQuarter             : (T,) bool — True at quarter-end months
    rng                      : np.random.Generator
    Pt_prev                  : (6,6) ndarray or None
        Warm-start: pass back the Pt_final from the previous Gibbs iteration
        to skip the Lyapunov solve.  None on first call triggers the solve.

    Returns
    -------
    loglh    : float
        Log-likelihood Σ_t log p(y_t | F_{t-1}), used in the MH step (eq. 19).
    z_t      : (Tstar+1, 3) ndarray
        Columns = [draw, filtered mean, predicted mean] of z_t path.
        Row 0 and row 1 are prepended z_{t-1} lag values from the filter.
    Pt_final : (6,6) ndarray
        Final filtered z-factor covariance P_{T|T}; pass back as Pt_prev.
    """

    # Build state-space matrices for this Gibbs iteration
    coeff = get_coefficients_sv(
        yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs
    )

    F0_z        = coeff["F0_z"]
    H1_z_AL     = coeff["H1_z_AL"]
    H1_z_NL     = coeff["H1_z_NL"]
    Q_e_sel_AL  = coeff["Q_e_sel_AL"]
    Q_e_sel_NL  = coeff["Q_e_sel_NL"]
    Q_z         = coeff["Q_z"]
    phi_cc_val  = coeff["phi_cc"]
    Ystar_m     = coeff["Ystar_m"]
    Ystar_q     = coeff["Ystar_q"]
    nan_mask_m  = coeff["nan_mask_m"]
    nan_mask_q  = coeff["nan_mask_q"]
    Tstar       = coeff["Tstar"]

    is_qend = indexQuarter[:Tstar].astype(np.bool_)

    # ------------------------------------------------------------------
    # Initialise z-factor covariance Pt_zz (6×6).
    # Warm-start: reuse P_{T|T} from the previous Gibbs iteration to avoid
    # the Lyapunov solve on every call after the first.
    # ------------------------------------------------------------------
    if Pt_prev is not None:
        # Accept either (6,6) from this function or legacy (mdim,mdim) covariance
        Pt_zz = np.ascontiguousarray(
            Pt_prev if Pt_prev.shape == (6, 6) else Pt_prev[0:6, 0:6]
        )
    else:
        # First call: solve 6×6 discrete Lyapunov equation
        # P_zz = F1_zz P_zz F1_zz' + Q_zz  for the unconditional z-factor covariance
        F1_zz = np.zeros((6, 6)); F1_zz[0, 0] = phi_cc_val; F1_zz[1:6, 0:5] = np.eye(5)
        Q_zz  = np.zeros((6, 6)); Q_zz[0, 0]  = float(Q_z)   # only σ²_{z,0} on diagonal
        Pt_zz = solve_discrete_lyapunov(F1_zz, Q_zz)
        Pt_zz = 0.5 * (Pt_zz + Pt_zz.T)

    # Initial z-factor state mean: unconditional mean μ_z / (1 - φ_z) for z_t = 0
    phi_cc_val_f = float(phi_cc_val)
    At_z = np.mean(F0_z) / (1.0 - phi_cc_val_f + 1e-10) * np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float
    )

    # Pre-draw all standard normals for the Gibbs draws at once (one vectorised RNG call)
    randn_draw = rng.standard_normal((Tstar, 2))

    JITTER = 1e-9   # added to Pt_zz diagonal before sqrt to guard against numerical rounding

    # ------------------------------------------------------------------
    # Kalman filter loop: iterate eqs. (IA.13)–(IA.14) forward in time
    # ------------------------------------------------------------------
    if _NUMBA_AVAILABLE:
        z_d0, z_d1, z_f0, z_f1, z_p0, loglh, Pt_final = _kalman_loop_nb_opt(
            float(phi_cc_val),
            float(Q_z),
            np.ascontiguousarray(F0_z),
            np.ascontiguousarray(H1_z_AL),
            np.ascontiguousarray(H1_z_NL),
            np.ascontiguousarray(Q_e_sel_AL),
            np.ascontiguousarray(Q_e_sel_NL),
            np.ascontiguousarray(is_qend),
            np.ascontiguousarray(nan_mask_m),
            np.ascontiguousarray(nan_mask_q),
            np.ascontiguousarray(Ystar_m),
            np.ascontiguousarray(Ystar_q),
            np.ascontiguousarray(At_z),
            np.ascontiguousarray(Pt_zz),
            np.ascontiguousarray(randn_draw),
            JITTER,
        )
    else:
        z_d0, z_d1, z_f0, z_f1, z_p0, loglh, Pt_final = _kalman_loop_numpy_opt(
            float(phi_cc_val), float(Q_z), F0_z,
            H1_z_AL, H1_z_NL, Q_e_sel_AL, Q_e_sel_NL,
            is_qend, nan_mask_m, nan_mask_q,
            Ystar_m, Ystar_q, At_z, Pt_zz, randn_draw, JITTER,
        )

    # ------------------------------------------------------------------
    # Reconstruct z_t output to match the original (Tstar+1, 3) layout.
    #
    # _reconstruct(A) in the original code was:
    #   np.concatenate([A[1:3, 1][::-1], A[1:, 0]])
    #   = [A[2,1], A[1,1], A[1,0], A[2,0], ..., A[Tstar-1, 0]]
    #
    # A[:, 0] = z_t draw column; A[:, 1] = z_{t-1} lag draw column.
    # The two prepended values are the z_{t-1} lag draws at Kalman steps
    # t=2 and t=1 (reversed), providing the initial z_{-1} and z_0 values.
    # A[1:, 0] skips the t=0 entry, matching the original [1:] slicing.
    # ------------------------------------------------------------------
    def _build_col_draw(d0, d1):
        # Prepend lag draws at t=2 and t=1, then z_t draws from t=1 onward
        return np.concatenate([[d1[2], d1[1]], d0[1:]])

    def _build_col_mean(m0, m1):
        return np.concatenate([[m1[2], m1[1]], m0[1:]])

    z_t = np.column_stack([
        _build_col_draw(z_d0, z_d1),   # column 0: z_t Gibbs draw
        _build_col_mean(z_f0, z_f1),   # column 1: z_t filtered mean α̂_{t|t}
        _build_col_mean(z_p0, z_f1),   # column 2: z_t predicted mean α̂_{t|t-1}
    ])   # shape (Tstar+1, 3)

    return loglh, z_t, Pt_final
