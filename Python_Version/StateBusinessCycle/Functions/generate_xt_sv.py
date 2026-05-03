# -*- coding: utf-8 -*-

"""
generate_xt_sv.py
-----------------
Kalman filter and one-step smoother draw for the latent state x_t in the
state-space model defined by equations (IA.13)–(IA.14):

    Measurement:  y_t   = A_t (H_0 + H_1 x_t + v_t),   v_t ~ N(0, R)      (IA.13)
    Transition:   x_{t+1} = F_0(S_t, S_{t+1}) + F_1 x_t + ω_{t+1},
                            ω_{t+1} ~ N(0, Q(S_{t+1}))                     (IA.14)

The state vector x_t holds the common growth factor z_t and its lags, as well
as the idiosyncratic components e_{i,t} for all monthly and quarterly series.
Conditioning on the current parameter draws and the imputed regime path S_t, the
Kalman filter computes the filtered distribution p(x_t | y_{1:t}); a draw from
this distribution is then the Gibbs step for the latent state path.

Time-aggregation of quarterly observables is handled via the time-varying
selection matrix A_t (equation IA.13): at quarter-end months A_t = A_last
which applies the Mariano-Murasawa weights (eq. 17); at other months
A_t = A_NotLast which selects only the monthly rows.

Performance optimisations (numerical results unchanged):
    OPT-1  Warm-start Pt: pass Pt_prev from the previous Gibbs iteration to
           skip the discrete Lyapunov solve on subsequent calls.
    OPT-2  Time-first memory layout for Q_t and F0_t: transposing these arrays
           so Q_tf[t] and F0_tf[t] are contiguous rows speeds up the inner loop.
    Numba JIT: the Kalman loop is compiled on first use; falls back to NumPy.

Outputs
-------
loglh    : float          log-likelihood summed over t (used in MH step)
z_t      : (Tstar+1, 3)  columns = [draw, filtered, predicted] of z_t path
Pt_final : (mdim, mdim)  final filtered covariance — pass back as Pt_prev
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, cho_factor, cho_solve

try:
    from .get_coefficients_sv import get_coefficients_sv
except ImportError:
    from get_coefficients_sv import get_coefficients_sv

_LOG2PI = np.log(2.0 * np.pi)

# ---------------------------------------------------------------------------
# Numba kernel — compiled once, cached for the session
# ---------------------------------------------------------------------------
_NUMBA_AVAILABLE = False
try:
    from numba import njit as _njit

    @_njit
    def _chol_lower_nb(A):
        n = A.shape[0]; L = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                s = A[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    L[i, j] = np.sqrt(max(s, 1e-15))
                else:
                    L[i, j] = s / L[j, j]
        return L

    @_njit
    def _chol_solve_nb(A, b):
        """Solve A x = b (A PD) and return (x, log|A|)."""
        L = _chol_lower_nb(A)
        n = L.shape[0]; m = b.shape[1]
        y = np.zeros((n, m)); x = np.zeros((n, m)); logdet = 0.0
        for i in range(n):
            logdet += 2.0 * np.log(L[i, i])
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

    # OPT-2: Q_tf is (Tstar, mdim, mdim), F0_tf is (Tstar, mdim) — time-first
    # so Q_tf[t] and F0_tf[t] are contiguous in memory.
    @_njit
    def _kalman_loop_nb(
        F1, Q_tf, F0_tf, is_qend,   # Q_tf[t] = Q at step t, F0_tf[t] = F0 col
        H0_AL, H1_AL, RR_AL,         # (N_m+N_q, *)  end-of-quarter (A_last branch)
        H0_NL, H1_NL, RR_NL,         # (N_m, *)      other months (A_NotLast branch)
        Ym, Yq,                       # (Tstar, N_m), (Tstar, N_q) pre-whitened data
        At0, Pt0,                     # initial state (mdim,1), initial cov (mdim,mdim)
        randn_state,                  # (Tstar, mdim) pre-drawn N(0,I) for state draws
        jitter,
    ):
        Tstar = Ym.shape[0]; mdim = F1.shape[0]
        Nm = Ym.shape[1]; Nq = Yq.shape[1]
        n_last = H0_AL.shape[0]; n_not = H0_NL.shape[0]

        At_draw = np.zeros((Tstar, mdim))
        At_mat  = np.zeros((Tstar, mdim))
        At_pred = np.zeros((Tstar, mdim))
        loglh   = 0.0

        At = np.ascontiguousarray(At0)
        Pt = np.ascontiguousarray(Pt0)
        JITTER_diag = np.eye(mdim) * jitter

        for t in range(Tstar):

            # Select measurement branch based on quarter-end indicator:
            # A_last at quarter-end months (monthly + quarterly observables),
            # A_NotLast otherwise (monthly observables only) — eq. (IA.13)
            if is_qend[t]:
                H0_A = H0_AL; H1_A = H1_AL; RR_A = RR_AL; n = n_last
                y_t = np.empty((n, 1))
                for i in range(Nm):   y_t[i, 0]      = Ym[t, i]
                for i in range(Nq):   y_t[Nm + i, 0] = Yq[t, i]
            else:
                H0_A = H0_NL; H1_A = H1_NL; RR_A = RR_NL; n = n_not
                y_t = np.empty((n, 1))
                for i in range(n):    y_t[i, 0] = Ym[t, i]

            # NaN mask — count valid observations (handles missing data)
            nv = 0
            for i in range(n):
                if not np.isnan(y_t[i, 0]): nv += 1

            # Build sub-matrices restricted to non-missing rows
            H0_M = np.empty((nv, 1))
            H1_M = np.empty((nv, mdim))
            RR_M = np.empty((nv, nv))
            y_m  = np.empty((nv, 1))
            ri = 0
            for i in range(n):
                if not np.isnan(y_t[i, 0]):
                    H0_M[ri, 0] = H0_A[i, 0]
                    for c in range(mdim): H1_M[ri, c] = H1_A[i, c]
                    y_m[ri, 0] = y_t[i, 0]
                    rj = 0
                    for j in range(n):
                        if not np.isnan(y_t[j, 0]):
                            RR_M[ri, rj] = RR_A[i, j]
                            rj += 1
                    ri += 1

            # OPT-2: contiguous access — Q_tf[t] and F0_tf[t] are row-major
            F0_col   = np.ascontiguousarray(F0_tf[t]).reshape(mdim, 1)
            Q_col    = np.ascontiguousarray(Q_tf[t])

            # Prediction step (eq. IA.14):
            # α̂_{t|t-1} = F0_t + F1 * α̂_{t-1|t-1}
            # P_{t|t-1}  = F1 * P_{t-1|t-1} * F1' + Q_t
            alphahat = F0_col + F1 @ At
            Phat     = F1 @ Pt @ F1.T + Q_col
            Phat     = 0.5 * (Phat + Phat.T)

            # Innovation and its covariance (eq. IA.13):
            # ν_t = y_t - H0 - H1 * α̂_{t|t-1}
            # F_t = H1 * P_{t|t-1} * H1' + R
            nut = y_m - H0_M - H1_M @ alphahat
            Ft  = H1_M @ Phat @ H1_M.T + RR_M
            Ft  = 0.5 * (Ft + Ft.T)

            # Cholesky-based solve: F_t^{-1} ν_t and log|F_t|
            invFt_nut, logdet = _chol_solve_nb(Ft, nut)
            loglh += -0.5 * nv * np.log(2.0 * np.pi) - 0.5 * logdet \
                     - 0.5 * (nut.T @ invFt_nut)[0, 0]

            # Update step (Kalman gain K_t = P_{t|t-1} H1' F_t^{-1}):
            # α̂_{t|t} = α̂_{t|t-1} + K_t ν_t
            # P_{t|t}  = P_{t|t-1} - K_t H1 P_{t|t-1}
            Ph1 = Phat @ H1_M.T
            At  = alphahat + Ph1 @ invFt_nut
            invFt_H1P, _ = _chol_solve_nb(Ft, np.ascontiguousarray(H1_M @ Phat))
            Pt  = Phat - Ph1 @ invFt_H1P
            Pt  = 0.5 * (Pt + Pt.T)

            for c in range(mdim):
                At_mat[t, c]  = At[c, 0]
                At_pred[t, c] = alphahat[c, 0]

            # Gibbs draw: x_t ~ N(α̂_{t|t}, P_{t|t})
            # Adding jitter to diagonal prevents Cholesky failure from floating-point rounding
            L    = _chol_lower_nb(Pt + JITTER_diag)
            z    = np.ascontiguousarray(randn_state[t, :]).reshape(mdim, 1)
            draw = (L @ z).reshape(mdim)
            for c in range(mdim):
                At_draw[t, c] = At[c, 0] + draw[c]

        # Return final Pt for OPT-1 warm-start on the next Gibbs iteration
        return At_draw, At_mat, At_pred, loglh, Pt

    # Trigger JIT compilation on a tiny problem
    def _warmup_numba():
        _m = 5; _n = 3; _T = 4
        _F1  = np.eye(_m)
        # OPT-2: time-first shapes for warmup
        _Q_tf  = np.stack([np.eye(_m) * 0.01] * _T, axis=0)   # (T, m, m)
        _F0_tf = np.zeros((_T, _m))                            # (T, m)
        _iq  = np.zeros(_T, dtype=np.bool_); _iq[2] = True
        _H0L = np.zeros((_n + 1, 1)); _H1L = np.random.randn(_n + 1, _m)
        _RRL = np.eye(_n + 1) * 0.1
        _H0N = np.zeros((_n, 1));     _H1N = np.random.randn(_n, _m)
        _RRN = np.eye(_n) * 0.1
        _Ym  = np.random.randn(_T, _n)
        _Yq  = np.random.randn(_T, 1)
        _At0 = np.zeros((_m, 1)); _Pt0 = np.eye(_m)
        _rdn = np.random.randn(_T, _m)
        _kalman_loop_nb(_F1, _Q_tf, _F0_tf, _iq, _H0L, _H1L, _RRL,
                        _H0N, _H1N, _RRN, _Ym, _Yq, _At0, _Pt0, _rdn, 1e-9)

    _warmup_numba()
    _NUMBA_AVAILABLE = True

except Exception:
    pass


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def generate_xt_sv(
    yy_monthly,
    yy_quarterly,
    s_t,
    param_macro_MH,
    param_macro_gibbs,
    indexQuarter,
    rng,
    Pt_prev=None,       # OPT-1: warm-start covariance from previous call
):
    """
    Parameters
    ----------
    yy_monthly   : (T, N_m) ndarray
    yy_quarterly : (T, N_q) ndarray
    s_t          : (T,) ndarray   0=recession, 1=expansion
    param_macro_MH   : dict
    param_macro_gibbs : dict
    indexQuarter : (T,) int ndarray   1 at end-of-quarter months
    rng : np.random.Generator
    Pt_prev : (mdim, mdim) ndarray or None
        If provided, skip solve_discrete_lyapunov and use this as the
        initial Kalman covariance (warm-start, OPT-1).

    Returns
    -------
    loglh    : float
    z_t      : (Tstar+1, 3) ndarray  columns = [draw, filtered, predicted]
    Pt_final : (mdim, mdim) ndarray  — pass back as Pt_prev next call
    """

    # ------------------------------------------------------------------
    # Build state-space matrices from eqs. (IA.13)–(IA.14)
    # ------------------------------------------------------------------
    (Ystar, H0, H1, RR, F0_t, F1, Q_t,
     A_select, Ystar_m, Ystar_q) = get_coefficients_sv(
        yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs
    )

    A_last    = A_select["A_last"]
    A_NotLast = A_select["A_NotLast"]

    Tstar = Ystar.shape[0]
    mdim  = F1.shape[0]

    # ------------------------------------------------------------------
    # Precompute A_t·H products for both measurement branches (eq. IA.13):
    #   At quarter-end:   y_t = A_last  * (H0 + H1*x_t)
    #   At other months:  y_t = A_NotLast * (H0 + H1*x_t)
    # ------------------------------------------------------------------
    H0_AL = np.ascontiguousarray(A_last @ H0)
    H1_AL = np.ascontiguousarray(A_last @ H1)
    RR_AL = np.ascontiguousarray(A_last @ RR @ A_last.T)

    H0_NL = np.ascontiguousarray(A_NotLast @ H0)
    H1_NL = np.ascontiguousarray(A_NotLast @ H1)
    RR_NL = np.ascontiguousarray(A_NotLast @ RR @ A_NotLast.T)

    F1c  = np.ascontiguousarray(F1)

    # ------------------------------------------------------------------
    # OPT-1: Initialise Pt — skip Lyapunov solve if warm-start provided.
    # On first call, solve P = F1 P F1' + Q_bar for the unconditional
    # covariance as the Kalman initialisation.
    # ------------------------------------------------------------------
    if Pt_prev is not None:
        Pt = np.ascontiguousarray(Pt_prev)
    else:
        Q_bar = np.mean(Q_t, axis=2)
        Pt    = solve_discrete_lyapunov(F1c, Q_bar)
        Pt    = 0.5 * (Pt + Pt.T)

    At = np.mean(F0_t, axis=1).reshape(mdim, 1)

    is_qend = indexQuarter[:Tstar].astype(np.bool_)

    # Pre-draw all standard normals for Gibbs state draws x_t ~ N(α̂_{t|t}, P_{t|t})
    randn_state = rng.standard_normal((Tstar, mdim))

    JITTER = 1e-9

    # ------------------------------------------------------------------
    # OPT-2: Transpose Q_t / F0_t to time-first for contiguous access.
    # Q_t  was (mdim, mdim, Tstar) → Q_tf  is (Tstar, mdim, mdim)
    # F0_t was (mdim, Tstar)       → F0_tf is (Tstar, mdim)
    # ------------------------------------------------------------------
    Q_tf  = np.ascontiguousarray(np.transpose(Q_t,  (2, 0, 1)))  # (Tstar, mdim, mdim)
    F0_tf = np.ascontiguousarray(F0_t.T)                          # (Tstar, mdim)

    # ------------------------------------------------------------------
    # Kalman filter loop: iterate eqs. (IA.13)–(IA.14) forward in time
    # ------------------------------------------------------------------
    if _NUMBA_AVAILABLE:
        At_draw, At_mat, At_pred, loglh, Pt_final = _kalman_loop_nb(
            F1c, Q_tf, F0_tf, is_qend,
            H0_AL, H1_AL, RR_AL,
            H0_NL, H1_NL, RR_NL,
            np.ascontiguousarray(Ystar_m),
            np.ascontiguousarray(Ystar_q),
            np.ascontiguousarray(At),
            np.ascontiguousarray(Pt),
            randn_state,
            JITTER,
        )
    else:
        At_draw, At_mat, At_pred, loglh, Pt_final = _kalman_loop_numpy(
            F1c, F1c.T, Q_tf, F0_tf, is_qend,
            H0_AL, H1_AL, RR_AL,
            H0_NL, H1_NL, RR_NL,
            Ystar_m, Ystar_q,
            At, Pt,
            randn_state,
            JITTER,
        )

    # ------------------------------------------------------------------
    # Reconstruct the z_t path from the state vector.
    # The state vector stores [z_t, z_{t-1}, ..., z_{t-5}, e_{m1,t}, ...].
    # Column 0 = z_t, column 1 = z_{t-1}.  Prepend the two initial lags
    # (rows 1 and 2 from the second column) to recover a continuous z series.
    # ------------------------------------------------------------------
    def _reconstruct(A):
        return np.concatenate([A[1:3, 1][::-1], A[1:, 0]])

    z_t = np.column_stack([
        _reconstruct(At_draw),
        _reconstruct(At_mat),
        _reconstruct(At_pred),
    ])

    return loglh, z_t, Pt_final


# ---------------------------------------------------------------------------
# NumPy fallback (used when numba is unavailable)
# OPT-2 applied: accepts Q_tf (Tstar,mdim,mdim) and F0_tf (Tstar,mdim)
# ---------------------------------------------------------------------------

def _kalman_loop_numpy(
    F1, F1T, Q_tf, F0_tf, is_qend,   # OPT-2: time-first arrays
    H0_AL, H1_AL, RR_AL,
    H0_NL, H1_NL, RR_NL,
    Ystar_m, Ystar_q,
    At, Pt,
    randn_state,
    jitter,
):
    Tstar = Ystar_m.shape[0]; mdim = F1.shape[0]
    JITTER_mat = jitter * np.eye(mdim)

    At_draw = np.zeros((Tstar, mdim))
    At_mat  = np.zeros((Tstar, mdim))
    At_pred = np.zeros((Tstar, mdim))
    loglh   = 0.0

    for t in range(Tstar):

        # Select measurement branch: A_last at quarter-end, A_NotLast otherwise
        if is_qend[t]:
            H0_A = H0_AL; H1_A = H1_AL; RR_A = RR_AL
            y_t_full = np.concatenate([Ystar_m[t, :], Ystar_q[t, :]])
        else:
            H0_A = H0_NL; H1_A = H1_NL; RR_A = RR_NL
            y_t_full = Ystar_m[t, :]

        # Drop NaN rows (handles unbalanced panel and missing observations)
        mask    = ~np.isnan(y_t_full)
        nv      = int(mask.sum())
        H0_M    = H0_A[mask, :]
        H1_M    = H1_A[mask, :]
        RR_M    = RR_A[np.ix_(mask, mask)]
        y_t     = y_t_full[mask].reshape(-1, 1)

        # OPT-2: contiguous row access
        F0_col   = F0_tf[t].reshape(mdim, 1)    # regime-dependent intercept F0(S_t,S_{t+1}) from eq. (IA.14);
        # only position 0 (z_t) is non-zero — lag slots and idiosyncratic components have no intercept.
        Q_col    = Q_tf[t]                      # regime-dependent noise covariance Q(S_{t+1}) from eq. (IA.14)

        # Prediction step: propagate state and covariance through eq. (IA.14)
        alphahat = F0_col + F1 @ At             # α̂_{t|t-1} = F0_t + F1 * α̂_{t-1|t-1}
        Phat     = F1 @ Pt @ F1T + Q_col        # P_{t|t-1} = F1 P_{t-1|t-1} F1' + Q(S_{t+1})
        Phat     = 0.5 * (Phat + Phat.T)        # symmetrise for numerical stability

        # Innovation and its covariance: residual between observed and predicted via eq. (IA.13)
        nut = y_t - H0_M - H1_M @ alphahat      # ν_t = y_t - H0 - H1 * α̂_{t|t-1}
        Ft  = H1_M @ Phat @ H1_M.T + RR_M       # F_t = H1 P_{t|t-1} H1' + R (innovation covariance)
        Ft  = 0.5 * (Ft + Ft.T)                 # symmetrise

        # Cholesky factorisation of F_t for stable inversion and log-determinant
        Ft_c    = cho_factor(Ft)
        logdet  = 2.0 * np.sum(np.log(np.abs(np.diag(Ft_c[0]))))
        invFt_nut = cho_solve(Ft_c, nut)        # F_t^{-1} ν_t via triangular solve

        # Log-likelihood contribution: log N(ν_t; 0, F_t)
        loglh += (-0.5 * nv * _LOG2PI - 0.5 * logdet
                  - 0.5 * float((nut.T @ invFt_nut).item()))

        # Update step: Kalman gain K_t = P_{t|t-1} H1' F_t^{-1}
        Ph1 = Phat @ H1_M.T                     # P_{t|t-1} H1': numerator of Kalman gain
        At  = alphahat + Ph1 @ invFt_nut        # α̂_{t|t} = α̂_{t|t-1} + K_t ν_t
        Pt  = Phat - Ph1 @ cho_solve(Ft_c, H1_M @ Phat)   # P_{t|t} = P_{t|t-1} - K_t H1 P_{t|t-1}
        Pt  = 0.5 * (Pt + Pt.T)                 # symmetrise

        At_mat[t, :]  = At.flatten()            # store filtered mean α̂_{t|t}
        At_pred[t, :] = alphahat.flatten()      # store predicted mean α̂_{t|t-1}

        # Gibbs draw: x_t ~ N(α̂_{t|t}, P_{t|t}).
        # A small jitter (1e-9 I) is added to P_{t|t} before Cholesky to guard against
        # near-singular covariances from floating-point accumulation.
        # If x = μ + L ε with ε ~ N(0,I), then Var(x) = L L' = P_{t|t} (since L is Cholesky).
        L = np.linalg.cholesky(Pt + JITTER_mat)
        At_draw[t, :] = At.flatten() + L @ randn_state[t, :]

    return At_draw, At_mat, At_pred, loglh, Pt
