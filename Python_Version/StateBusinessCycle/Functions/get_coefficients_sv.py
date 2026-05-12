# -*- coding: utf-8 -*-

"""
get_coefficients_sv.py
----------------------
Builds the compact state-space representation for the macroeconomic block
defined by equations (IA.13)–(IA.16) of the Internet Appendix:

    Measurement:  y_t   = A_t (H0 + H1 * x_t + e_t),   Var(e_t) = RR   (IA.13)
    Transition:   x_t   = F0_t + F1 * x_{t-1} + v_t,   Var(v_t) = Q_t  (IA.14)

The observables y_t are the pre-whitened macro series y*_{i,t} (see below).
A_t is a time-varying selection matrix: at quarter-end months it selects both
monthly and quarterly rows (A_last); at other months only monthly rows (A_NotLast).

State vector x_{t+1} (length nDim = 6 + N_m + N_q*tau):
    [z_t, z_{t-1}, z_{t-2}, z_{t-3}, z_{t-4}, z_{t-5},   <- 6 lags of common growth z
     e_m1_t, ..., e_mNm_t,                                 <- N_m monthly idiosyncratic shocks
     e_q1_t, e_q1_{t-1}, ..., e_q1_{t-4},                 <- 5 lags for quarterly variable 1
     ...
     e_qNq_t, ..., e_qNq_{t-4}]                            <- 5 lags for quarterly variable Nq

Key design notes:
  - Equation (13): each observable loads on the common growth factor z_t via
    γ_i and on its own idiosyncratic component e_{i,t}.
  - Equation (15): e_{i,t} follows an AR(1) with coefficient ψ_i.  The AR(1)
    structure is absorbed into H1 and the pre-whitened data Ystar; F1 only
    contains the companion form for the 6 z-lags.  F1 rows 6+ are therefore
    zero: the idiosyncratic states are driven entirely by their noise Q_i
    and the Kalman update, with no AR carry-over through F1.
  - Equation (17) / (IA.16): quarterly observables are time-aggregated monthly
    values with Mariano–Murasawa weights [1,2,3,2,1]/3, encoded in A_last
    and in the z-loading block H1_macro_q.
  - Quarterly AR(1) pre-whitening uses a lag-3 difference (quarterly frequency)
    to match the observed quarterly growth rate.
  - Ystar_m drops 2 extra rows so both Ystar_m and Ystar_q have T-3 rows,
    giving Tstar = T-3.
  - F0_t and Q_t are built at full length then trimmed to Tstar columns/slices.

What is returned
----------------
The Kalman filter in generate_xt_sv maintains only the 6-dim z-factor state
[z_t, z_{t-1}, ..., z_{t-5}] and its 6×6 covariance Pt_zz.  This is valid
because F1 rows 6+ are zero, so at the prediction step:
    Phat = F1 @ Pt @ F1.T + Q
is non-zero only in the 6×6 z-factor block (rows/cols 0–5).  The idiosyncratic
block resets to diag(SIG2_i) at every prediction step — their contribution to
the innovation covariance Ft is therefore the diagonal term
    H1_e @ diag(SIG2_i) @ H1_e.T = diag(Q_e_sel)
where Q_e_sel[i] is the idiosyncratic variance for observable i.

This function therefore returns:
    H1_z_AL, H1_z_NL : z-factor columns of A_t @ H1   (eq. IA.13)
    Q_e_sel_AL, _NL  : per-observable idiosyncratic Q  (diagonal of H1_e @ Phat_ee @ H1_e.T)
    Q_z              : common-factor noise σ²_{z,0}    (= Sigma2_0_cc, hardcoded to 1.0)
    F0_z             : z-factor row of F0_t             (μ_z(S_{t+1}) - φ_z μ_z(S_t), eq. 14)
    Ystar_m, Ystar_q : pre-whitened data                (eq. IA.15)
    nan_mask_m, _q   : True = valid (non-NaN) observation at each (t, variable)
"""

import numpy as np


def get_coefficients_sv(
    yy_monthly,
    yy_quarterly,
    s_t,
    param_macro_MH,
    param_macro_gibbs,
):
    """
    Parameters
    ----------
    yy_monthly   : (T, N_m) ndarray  – standardised monthly data (NaNs allowed)
    yy_quarterly : (T, N_q) ndarray  – standardised quarterly data (NaNs allowed)
    s_t          : (T,) ndarray      – regime indicator (0=expansion, 1=recession)
    param_macro_MH : dict
        Keys: 'paramMU'      – (2,) array [mu_0, mu_1]
              'Sigma2_0_cc'  – float  (σ²_{z,0}, always 1.0)
              'h_cc'         – float  (h_z, always 0.0)
              'phi_cc'       – float  (φ_z, AR(1) persistence of z_t)
    param_macro_gibbs : dict
        Keys: 'gamma_macro_m'   – (N_m+3,) array  [N_m-1 scalars, then 4 for last var]
              'psi_macro_m'     – (N_m,) array
              'SIG2_i_macro_m'  – (N_m,) array
              'gamma_macro_q'   – (N_q,) array
              'psi_macro_q'     – (N_q,) array
              'SIG2_i_macro_q'  – (N_q,) array

    Returns
    -------
    dict with keys:
        'F0_z'        : (Tstar,)       z-factor intercept F0_t[0, :] from eq. (14)
        'H1_z_AL'     : (N_m+N_q, 6)  z-loadings A_last @ H1[:, 0:6]  (eq. IA.13)
        'H1_z_NL'     : (N_m,     6)  z-loadings A_NotLast @ H1[:, 0:6]
        'Q_e_sel_AL'  : (N_m+N_q,)    idiosyncratic Q per observable, quarter-end
        'Q_e_sel_NL'  : (N_m,)        idiosyncratic Q per observable, non-quarter-end
        'Q_z'         : float          σ²_{z,0} = Sigma2_0_cc (hardcoded 1.0)
        'phi_cc'      : float          φ_z
        'Ystar_m'     : (Tstar, N_m)   pre-whitened monthly data  (eq. IA.15)
        'Ystar_q'     : (Tstar, N_q)   pre-whitened quarterly data
        'nan_mask_m'  : (Tstar, N_m)   bool, True = valid observation
        'nan_mask_q'  : (Tstar, N_q)   bool
        'Tstar'       : int            T - 3
        'mdim'        : int            6 + N_m + N_q*tau
        'N_m'         : int
        'N_q'         : int
    """

    yy_monthly   = np.asarray(yy_monthly,   dtype=float)
    yy_quarterly = np.asarray(yy_quarterly, dtype=float)
    s_t          = np.asarray(s_t,          dtype=float).reshape(-1)

    T,   N_m = yy_monthly.shape
    _,   N_q = yy_quarterly.shape

    tau = 5  # monthly periods that aggregate to one quarter; weights from eq. (17) / (IA.16):
             # Δy^{qrt}_{i,t+1} = Σ_{j=1}^{5} (3-|j-3|)/3 * Δy_{i,t+2-j}

    # ------------------------------------------------------------------
    # Unpack parameters
    # ------------------------------------------------------------------
    gamma_macro_m  = np.asarray(param_macro_gibbs["gamma_macro_m"],  dtype=float).reshape(-1)
    psi_macro_m    = np.asarray(param_macro_gibbs["psi_macro_m"],    dtype=float).reshape(-1)
    SIG2_i_macro_m = np.asarray(param_macro_gibbs["SIG2_i_macro_m"], dtype=float).reshape(-1)

    gamma_macro_q  = np.asarray(param_macro_gibbs["gamma_macro_q"],  dtype=float).reshape(-1)
    psi_macro_q    = np.asarray(param_macro_gibbs["psi_macro_q"],    dtype=float).reshape(-1)
    SIG2_i_macro_q = np.asarray(param_macro_gibbs["SIG2_i_macro_q"], dtype=float).reshape(-1)

    paramMU      = np.asarray(param_macro_MH["paramMU"], dtype=float).reshape(-1)
    Sigma2_0_cc  = float(param_macro_MH["Sigma2_0_cc"])   # σ²_{z,0}: baseline variance (always 1.0)
    h_cc         = float(param_macro_MH["h_cc"])           # h_z: regime volatility shifter (always 0.0)
    phi_cc       = float(param_macro_MH["phi_cc"])         # φ_z: AR(1) persistence of z_t (eq. 14)

    mu_0, mu_1 = float(paramMU[0]), float(paramMU[1])     # μ_0, μ_1: regime means (eq. 14)

    tau_aux = np.array([1, 2, 3, 2, 1], dtype=float) / 3.0  # (5,) Mariano–Murasawa weights (eq. 17)
    mdim    = 6 + N_m + N_q * tau                            # total state-vector dimension

    # ------------------------------------------------------------------
    # z-factor loadings H1_z — the first 6 columns of H1 (eq. IA.13)
    # ------------------------------------------------------------------
    # From eq. (IA.15): y*_{i,t+1} = γ_i z_{t+1} - γ_i ψ_i z_t + σ_i ε_{i,t+1}
    # This encodes the pre-whitened form of eq. (13) after applying ψ_i to remove
    # the AR(1) in e_{i,t} (eq. 15).
    #
    # Monthly: gamma_macro_m has length N_m+3:
    #   gamma_f = gamma_macro_m[:-4]  shape (N_m-1,)  scalar loadings for first N_m-1 variables
    #   gamma_l = gamma_macro_m[-4:]  shape (4,)       4-lag loadings for last variable
    #
    # For first N_m-1 variables, z-block row i is:
    #   [γ_i, -γ_i*ψ_i, 0, 0, 0, 0]   (from IA.15: coefficient on z_t and z_{t-1})
    #
    # For last variable (weekly jobless claims, aggregated to monthly with 4 lags):
    #   [γ_0, γ_1-ψ*γ_0, γ_2-ψ*γ_1, γ_3-ψ*γ_2, -ψ*γ_3, 0]

    gamma_f = gamma_macro_m[:-4]          # (N_m-1,) scalar loadings
    gamma_l = gamma_macro_m[-4:]          # (4,) 4-lag loadings for last variable
    psi_f   = psi_macro_m[:-1]            # (N_m-1,)
    psi_l   = float(psi_macro_m[-1])

    # (N_m-1, 6) z-block for first N_m-1 variables: [γ, -γψ, 0, 0, 0, 0]
    H1_z_m_head = np.zeros((N_m - 1, 6))
    H1_z_m_head[:, 0] =  gamma_f
    H1_z_m_head[:, 1] = -gamma_f * psi_f

    # (1, 6) z-block for last variable (4-lag pre-whitened form of eq. IA.15)
    H1_z_m_last = np.array([[
        gamma_l[0],
        gamma_l[1] - psi_l * gamma_l[0],
        gamma_l[2] - psi_l * gamma_l[1],
        gamma_l[3] - psi_l * gamma_l[2],
                   - psi_l * gamma_l[3],
        0.0,
    ]])

    H1_z_NL = np.vstack([H1_z_m_head, H1_z_m_last])    # (N_m, 6) — z-loadings for monthly observables

    # ---- Quarterly z-loadings: apply Mariano–Murasawa weights (eq. IA.16) ----
    #
    # Encodes the quarterly aggregation (eq. 17 / IA.16) for each quarterly variable j.
    # For the 5-row companion block of quarterly variable j:
    #   mat_aux1[ii, ii]   = γ_j          (diagonal: z loadings at each monthly lag)
    #   mat_aux1[jj, jj+1] = -γ_j * ψ_j  (superdiag: pre-whitened AR(1) correction)
    # Applying tau_aux (1×5) @ (5×6) contracts the 5 monthly rows to one quarterly row.

    gamma_psi_q = -gamma_macro_q * psi_macro_q    # (N_q,)
    H1_z_q = np.zeros((N_q, 6))
    for j in range(N_q):
        g  = float(gamma_macro_q[j])
        gp = float(gamma_psi_q[j])
        for ii in range(5):                # γ_j on diagonal (z loadings at each monthly lag)
            H1_z_q[j, ii] += tau_aux[ii] * g
        for jj in range(5):                # -γ_j*ψ_j on superdiagonal (pre-whitened AR(1) correction)
            if jj + 1 < 6:
                H1_z_q[j, jj + 1] += tau_aux[jj] * gp

    # Quarter-end months observe monthly + quarterly variables
    H1_z_AL = np.vstack([H1_z_NL, H1_z_q])              # (N_m+N_q, 6)

    # ------------------------------------------------------------------
    # Idiosyncratic noise contribution per observable — Q_e_sel (eq. IA.13)
    # ------------------------------------------------------------------
    # At the prediction step, F1 rows 6+ are zero so Phat_ee = diag(SIG2_i)
    # (the idiosyncratic prediction covariance resets to Q each step).
    # H1_e selects exactly one idiosyncratic state per observable:
    #
    # Monthly observation i:   H1_e[i, i]=1  → Q_e_sel_NL[i] = SIG2_m[i]
    # Quarterly observation j at quarter-end: H1_e applies tau_aux weights
    #   across 5 lag slots where only the first has non-zero Q (= SIG2_q[j]):
    #   Q_e_sel_AL[N_m+j] = tau_aux[0]² * SIG2_q[j] = SIG2_q[j] / 9
    Q_e_sel_NL = SIG2_i_macro_m.copy()                     # (N_m,)

    Q_e_sel_q  = (tau_aux[0] ** 2) * SIG2_i_macro_q        # (N_q,) = SIG2_q / 9
    Q_e_sel_AL = np.concatenate([SIG2_i_macro_m, Q_e_sel_q])  # (N_m+N_q,)

    # ------------------------------------------------------------------
    # F0_t — time-varying intercept, z-factor row only (eq. IA.14)
    # ------------------------------------------------------------------
    # From eq. (14): z_{t+1} = μ_z(S_{t+1}) + φ_z(z_t - μ_z(S_t)) + σ_z ε
    # Rearranging: F0_t[0, t] = μ_z(S_{t+1}) - φ_z * μ_z(S_t)
    # where μ_z(S_t) = μ_0 + μ_1 * S_t  (regime-dependent mean, eq. 14).
    # All other rows of F0_t are zero: lag slots and idiosyncratic components
    # have no intercept.
    mu_aux  = mu_0 + mu_1 * s_t                     # (T,), μ_z(S_t) = μ_0 + μ_1*S_t
    mu_t    = mu_aux[1:] - phi_cc * mu_aux[:-1]      # (T-1,), F0_t first row

    Tstar  = T - 3
    F0_z   = mu_t[-Tstar:]                           # (Tstar,) trimmed to align with Ystar

    # ------------------------------------------------------------------
    # Pre-whiten data to form Ystar (eq. IA.15)
    # ------------------------------------------------------------------
    # From eq. (IA.15): y*_{i,t+1} = y_{i,t+1} - ψ_i * y_{i,t}
    # This removes the AR(1) in e_{i,t} (eq. 15), yielding i.i.d. errors.
    #
    # Monthly:   y*_{i,t} = y_{i,t+1} - ψ_i * y_{i,t}    (lag-1 difference)
    #            Drop 2 extra rows so Ystar_m has Tstar = T-3 rows.
    #
    # Quarterly: y*_{i,t} = y_{i,t+3} - ψ_i * y_{i,t}    (lag-3 = one quarter)
    #            Already T-3 rows.
    yy_star_m = yy_monthly[1:,  :] - psi_macro_m[np.newaxis, :] * yy_monthly[:-1,  :]   # (T-1, N_m)
    yy_star_q = yy_quarterly[3:, :] - psi_macro_q[np.newaxis, :] * yy_quarterly[:-3, :]  # (T-3, N_q)

    Ystar_m = yy_star_m[2:, :]    # (Tstar, N_m) trim 2 rows to align with quarterly data
    Ystar_q = yy_star_q            # (Tstar, N_q)

    # ------------------------------------------------------------------
    # NaN masks — True where the pre-whitened observation is valid (non-NaN)
    # ------------------------------------------------------------------
    # NaN propagates from the raw data through the pre-whitening: y*_{i,t} is NaN
    # if either y_{i,t+1} or y_{i,t} is NaN.  The mask is therefore determined
    # by the fixed NaN structure of yy_monthly and yy_quarterly.
    nan_mask_m = ~np.isnan(Ystar_m)    # (Tstar, N_m)  True = valid
    nan_mask_q = ~np.isnan(Ystar_q)    # (Tstar, N_q)

    return {
        "F0_z"       : np.ascontiguousarray(F0_z,        dtype=float),
        "H1_z_AL"    : np.ascontiguousarray(H1_z_AL,     dtype=float),   # (N_m+N_q, 6)
        "H1_z_NL"    : np.ascontiguousarray(H1_z_NL,     dtype=float),   # (N_m,     6)
        "Q_e_sel_AL" : np.ascontiguousarray(Q_e_sel_AL,  dtype=float),   # (N_m+N_q,)
        "Q_e_sel_NL" : np.ascontiguousarray(Q_e_sel_NL,  dtype=float),   # (N_m,)
        "Q_z"        : float(Sigma2_0_cc),                                # σ²_{z,0} = 1.0
        "phi_cc"     : float(phi_cc),
        "Ystar_m"    : np.ascontiguousarray(Ystar_m,     dtype=float),
        "Ystar_q"    : np.ascontiguousarray(Ystar_q,     dtype=float),
        "nan_mask_m" : np.ascontiguousarray(nan_mask_m),
        "nan_mask_q" : np.ascontiguousarray(nan_mask_q),
        "Tstar"      : int(Tstar),
        "mdim"       : int(mdim),
        "N_m"        : int(N_m),
        "N_q"        : int(N_q),
    }
