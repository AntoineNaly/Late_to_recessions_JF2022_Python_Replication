# -*- coding: utf-8 -*-

"""
get_coefficients_sv.py
----------------------
Builds the state-space representation for the macroeconomic block defined by
equations (IA.13)–(IA.16) of the Internet Appendix:

    Measurement:  y_t   = A_t (H0 + H1 * x_t + e_t),   Var(e_t) = RR   (IA.13)
    Transition:   x_t   = F0_t + F1 * x_{t-1} + v_t,   Var(v_t) = Q_t  (IA.14)

The observables y_t are the pre-whitened macro series y*_{i,t} (see below).
A_t is a time-varying selection matrix that accounts for which variables are
observable at time t: at quarter-end months it selects both monthly and
quarterly rows (A_last); at other months it selects only monthly rows (A_NotLast).

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
    contains the companion form for the 6 z-lags.
  - Equation (17) / (IA.16): quarterly observables are time-aggregated monthly
    values with Mariano–Murasawa weights [1,2,3,2,1]/3.  This is encoded in
    the kron block of A_last and in H1_macro_q.
  - Quarterly AR(1) pre-whitening uses a lag-3 difference (quarterly frequency)
    to match the observed quarterly growth rate.
  - Ystar_m drops 2 extra rows so both Ystar_m and Ystar_q have T-3 rows,
    giving Tstar = T-3.
  - F0_t and Q_t are built at full length then trimmed to Tstar columns/slices.
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
              'Sigma2_0_cc'  – float
              'h_cc'         – float
              'phi_cc'       – float
    param_macro_gibbs : dict
        Keys: 'gamma_macro_m'   – (N_m+3,) array  [first N_m-1 scalars, then 4 for last var]
              'psi_macro_m'     – (N_m,) array
              'SIG2_i_macro_m'  – (N_m,) array
              'gamma_macro_q'   – (N_q,) array
              'psi_macro_q'     – (N_q,) array
              'SIG2_i_macro_q'  – (N_q,) array

    Returns
    -------
    Ystar      : (Tstar, N_m+N_q) ndarray   pre-whitened stacked observations
    H0         : (nStates_Y, 1) ndarray
    H1         : (nStates_Y, nDim) ndarray
    RR         : (nStates_Y, nStates_Y) ndarray  (all zeros)
    F0_t       : (nDim, Tstar) ndarray           time-varying transition intercept
    F1         : (nDim, nDim) ndarray            transition matrix
    Q_t        : (nDim, nDim, Tstar) ndarray     time-varying innovation covariance
    A_select   : dict  {'A_last': ..., 'A_NotLast': ...}
    Ystar_m    : (Tstar, N_m) ndarray
    Ystar_q    : (Tstar, N_q) ndarray
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
    Sigma2_0_cc  = float(param_macro_MH["Sigma2_0_cc"])
    h_cc         = float(param_macro_MH["h_cc"])
    phi_cc       = float(param_macro_MH["phi_cc"])

    mu_0, mu_1 = paramMU[0], paramMU[1]

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    tau_aux    = np.array([1, 2, 3, 2, 1], dtype=float) / 3.0  # (5,) Mariano-Murasawa weights from eq. (17)
    mDim       = N_m + N_q * tau + 6       # total state-vector dimension; 6 = tau+1 lags to capture AR(1) in e_{i,t}
    nStates_Y  = N_m + N_q * tau           # total observable dimension (rows of H1)
    nDim       = mDim

    # ------------------------------------------------------------------
    # Selection matrices A_last and A_NotLast (equation IA.13)
    # ------------------------------------------------------------------
    # A_last  selects all observables at quarter-end months (N_m monthly + N_q quarterly):
    #   top block:    [eye(N_m),  zeros(N_m, N_q*tau)]   <- monthly rows
    #   bottom block: [zeros(N_q, N_m),  kron(eye(N_q), tau_aux)]  <- quarterly aggregation (eq. IA.16)
    #
    # A_NotLast  selects only monthly observables at non-quarter-end months:
    #   [eye(N_m),  zeros(N_m, N_q*tau)]

    A_last = np.zeros((N_m + N_q, nStates_Y))
    A_last[:N_m, :N_m] = np.eye(N_m)
    # kron(eye(N_q), tau_aux) applies the Mariano-Murasawa weights (eq. 17) to each quarterly variable
    kron_block = np.kron(np.eye(N_q), tau_aux.reshape(1, -1))  # (N_q, N_q*tau)
    A_last[N_m:, N_m:] = kron_block

    A_NotLast = np.zeros((N_m, nStates_Y))
    A_NotLast[:N_m, :N_m] = np.eye(N_m)

    A_select = {"A_last": A_last, "A_NotLast": A_NotLast}

    # ------------------------------------------------------------------
    # H0  (nStates_Y, 1) — intercept in measurement equation (IA.13); zero
    # because the common factor z_t is already demeaned via F0_t
    # ------------------------------------------------------------------
    H0 = np.zeros((nStates_Y, 1))

    # ------------------------------------------------------------------
    # H1  (nStates_Y, nDim) — loadings in measurement equation (IA.13)
    # ------------------------------------------------------------------

    # ---- Monthly block: H1_macro_m  (N_m, nDim) ----
    #
    # From eq. (IA.15): y*_{i,t+1} = γ_i z_{t+1} - γ_i ψ_i z_t + σ_i ε_{i,t+1}
    # This encodes the pre-whitened form of eq. (13) after applying ψ_i to remove
    # the AR(1) in e_{i,t} (eq. 15).
    #
    # gamma_macro_m has length N_m+3:
    #   gamma_f = gamma_macro_m[:-4]   shape (N_m-1,)  loadings for first N_m-1 variables
    #   gamma_l = gamma_macro_m[-4:]   shape (4,)       4-lag loadings for last variable
    #
    # For first N_m-1 variables, row i of the 6-column z-block is:
    #   [γ_i, -γ_i*ψ_i, 0, 0, 0, 0]   (from IA.15: coefficient on z_t and z_{t-1})
    #
    # For last variable (weekly claims, aggregated to monthly with 4 lags):
    #   [γ_0, γ_1-ψ*γ_0, γ_2-ψ*γ_1, γ_3-ψ*γ_2, -ψ*γ_3, 0]
    #
    # H1_macro_m = [H1_macro_m_aux1 | eye(N_m) | zeros(N_m, N_q*tau)]
    # The identity block picks up e_{i,t} directly from the state vector.

    gamma_f  = gamma_macro_m[:-4]   # (N_m-1,) scalar loadings
    gamma_l  = gamma_macro_m[-4:]   # (4,) 4-lag loadings for last variable
    psi_1_f  = psi_macro_m[:-1]     # (N_m-1,)
    psi_1_l  = float(psi_macro_m[-1])

    # 6-column block for first N_m-1 variables: [γ, -γψ, 0, 0, 0, 0]
    H1_f = np.column_stack([gamma_f, -gamma_f * psi_1_f])  # (N_m-1, 2)
    H1_f_padded = np.hstack([H1_f, np.zeros((N_m - 1, 4))])  # (N_m-1, 6)

    # 6-column block for the last variable (4-lag pre-whitened form of eq. IA.15)
    gamma_0_l = gamma_l[0]
    gamma_1_l = gamma_l[1] - psi_1_l * gamma_l[0]
    gamma_2_l = gamma_l[2] - psi_1_l * gamma_l[1]
    gamma_3_l = gamma_l[3] - psi_1_l * gamma_l[2]
    gamma_4_l =            - psi_1_l * gamma_l[3]
    H1_l = np.array([[gamma_0_l, gamma_1_l, gamma_2_l, gamma_3_l, gamma_4_l, 0.0]])  # (1, 6)

    H1_macro_m_aux1 = np.vstack([H1_f_padded, H1_l])    # (N_m, 6)   z-block
    H1_macro_m_aux2 = np.eye(N_m)                       # (N_m, N_m) idiosyncratic block
    H1_macro_m = np.hstack([
        H1_macro_m_aux1,                                # columns 0-5:   z and its lags
        H1_macro_m_aux2,                                # columns 6-17:  identity picks up e_{i,t}
        np.zeros((N_m, N_q * tau)) ])                   # columns 18-32: zeros for quarterly states


    # ---- Quarterly block: H1_macro_q  (N_q*tau, nDim) ----
    #
    # Encodes the Mariano-Murasawa aggregation (eq. 17 / IA.16) for quarterly variables.
    # For each quarterly variable j, the pre-whitened aggregation of eq. (IA.15) gives
    # a 5-row block (one row per monthly period within the quarter):
    #   mat_aux1[i, i]   = γ_j                (diagonal:   coefficient on z_{t-i})
    #   mat_aux1[i, i+1] = -γ_j * ψ_j         (superdiag:  coefficient on z_{t-i-1})
    #
    # H1_macro_q = [H1_macro_q_1 | zeros(N_q*5, N_m) | kron(eye(N_q), eye(5))]

    gamma_psi = -gamma_macro_q * psi_macro_q  # (N_q,)
    auxVar    = np.column_stack([gamma_macro_q, gamma_psi])  # (N_q, 2)

    H1_macro_q_1_blocks = []
    for index_q in range(N_q):
        aux1     = auxVar[index_q]          # [γ_j, -γ_j*ψ_j]
        mat_aux1 = np.zeros((5, 6))
        for ii in range(5):                 # γ_j on diagonal (z loadings at each monthly lag)
            mat_aux1[ii, ii]     = aux1[0]
        for jj in range(5):                 # -γ_j*ψ_j on superdiagonal (pre-whitened AR(1) correction)
            mat_aux1[jj, jj + 1] = aux1[1]
        H1_macro_q_1_blocks.append(mat_aux1)

    H1_macro_q_1 = np.vstack(H1_macro_q_1_blocks)                    # (N_q*5, 6)
    H1_macro_q_2 = np.zeros((N_q * tau, N_m))                        # (N_q*5, N_m)
    H1_macro_q_3 = np.kron(np.eye(N_q), np.eye(tau))                 # (N_q*5, N_q*5)

    H1_macro_q = np.hstack([H1_macro_q_1, H1_macro_q_2, H1_macro_q_3])  # (N_q*5, nDim)

    H1 = np.vstack([H1_macro_m, H1_macro_q])  # (nStates_Y, nDim)

    # ------------------------------------------------------------------
    # RR  (nStates_Y, nStates_Y) — measurement noise covariance in (IA.13)
    # RR = 0 because both z_t and e_{i,t} are included as state variables,
    # so the measurement equation is exact (no additional noise term).
    # ------------------------------------------------------------------
    RR = np.zeros((nStates_Y, nStates_Y))

    # ------------------------------------------------------------------
    # F0_t  (nDim, Tstar) — time-varying intercept in transition eq. (IA.14)
    # ------------------------------------------------------------------
    # From eq. (14): z_{t+1} = μ_z(S_{t+1}) + φ_z(z_t - μ_z(S_t)) + σ_z(S_{t+1})ε_{z,t+1}
    # Rearranging for the state-space form: F0_t[0, t] = μ_z(S_{t+1}) - φ_z * μ_z(S_t)
    # where μ_z(S_t) = μ_0 + μ_1 * S_t  (regime-dependent mean from eq. 14).
    # All other rows of F0_t are zero: lag slots and idiosyncratic components have no intercept.

    mu_aux = mu_0 + mu_1 * s_t                          # (T,), μ_z(S_t) = μ_0 + μ_1*S_t
    mu_t   = mu_aux[1:] - phi_cc * mu_aux[:-1]          # (T-1,), F0_t first row: μ_z(S_{t+1}) - φ_z*μ_z(S_t)
    F0_t_full = np.zeros((mDim, T - 1))
    F0_t_full[0, :] = mu_t                              # only z's row is non-zero

    # ------------------------------------------------------------------
    # F1  (nDim, nDim) — transition matrix in eq. (IA.14)
    # ------------------------------------------------------------------
    # From eq. (14): z_{t+1} = ... + φ_z * (z_t - μ_z(S_t)) + ...
    # In state-space form, after absorbing the mean into F0_t:
    #   F1[0,0] = φ_z  (AR(1) persistence of the common factor)
    #   F1[1:6, 0:5] = eye(5)  (companion form: shift z_{t-k} lags forward)
    # Idiosyncratic AR(1) coefficients ψ_i are encoded in H1 and Ystar, not in F1.

    F1 = np.zeros((nDim, nDim))
    F1[0, 0] = phi_cc                          # AR(1) for z_t: φ_z in eq. (14)
    F1[1 : tau + 1, 0 : tau] = np.eye(tau)    # lag shifts for z_{t-k}

    # ------------------------------------------------------------------
    # Q_t  (nDim, nDim, Tstar) — time-varying innovation covariance in (IA.14)
    # ------------------------------------------------------------------
    # From eq. (14): σ²_z(S_t) = σ²_{z,0}*(1 + h_z*S_t)  (regime-dependent volatility)
    # Non-zero diagonal entries (0-indexed):
    #   [0, 0, t]          = σ²_{z,0}*(1 + h_z*S_t)    common factor variance (eq. 14)
    #   [6..6+N_m-1]       = σ²_{e,i}                   monthly idiosyncratic variance (eq. 15)
    #   [6+N_m, 6+N_m+5, ...]  = σ²_{e,j}              quarterly idiosyncratic variance (first lag only)

    Sigma2_t = Sigma2_0_cc * (1.0 + h_cc * s_t)  # (T,), σ²_z(S_t) from eq. (14)

    Q_t_full = np.zeros((nDim, nDim, T))
    Q_t_full[0, 0, :] = Sigma2_t

    # Monthly idiosyncratic variances σ²_{e,i} from eq. (15)
    for jjj, kk in enumerate(range(6, 6 + N_m)):
        Q_t_full[kk, kk, :] = SIG2_i_macro_m[jjj]

    # Quarterly idiosyncratic variances (first of the 5 lag slots per variable)
    for jjj, kk in enumerate(range(6 + N_m, 6 + N_m + N_q * 5, 5)):
        Q_t_full[kk, kk, :] = SIG2_i_macro_q[jjj]

    # ------------------------------------------------------------------
    # Pre-whiten data to form Ystar
    # ------------------------------------------------------------------
    # From eq. (IA.15): y*_{i,t+1} = y_{i,t+1} - ψ_i * y_{i,t}
    # This removes the AR(1) in e_{i,t} (eq. 15), yielding a regression with
    # i.i.d. errors that admits a clean conjugate posterior in the Gibbs sampler.
    #
    # Monthly:   y*_{i,t} = y_{i,t+1} - ψ_i * y_{i,t}         (lag-1 difference)
    #            Drop 2 extra rows so Ystar_m has Tstar = T-3 rows.
    #
    # Quarterly: y*_{i,t} = y_{i,t+3} - ψ_i * y_{i,t}         (lag-3 = one quarter)
    #            Already T-3 rows.

    psi_1_m = psi_macro_m  # (N_m,)
    psi_1_q = psi_macro_q  # (N_q,)
    # (T-1, N_m)
    yy_star_m = yy_monthly[1:, :] - psi_1_m[np.newaxis, :] * yy_monthly[:-1, :]
    # (T-3, N_q)
    yy_star_q = yy_quarterly[3:, :] - psi_1_q[np.newaxis, :] * yy_quarterly[:-3, :]

    Ystar_m = yy_star_m[2:, :]   # (T-3, N_m), trim 2 rows to align with quarterly data
    Ystar_q = yy_star_q           # (T-3, N_q)

    Ystar = np.hstack([Ystar_m, Ystar_q])  # (T-3, N_m+N_q)

    Tstar = Ystar.shape[0]   # T-3

    # ------------------------------------------------------------------
    # Trim F0_t and Q_t to last Tstar time slices
    # (aligns with the pre-whitened Ystar which starts at t=3)
    # ------------------------------------------------------------------
    F0_t  = F0_t_full[:, -Tstar:]           # (nDim, Tstar)
    Q_t   = Q_t_full[:, :, -Tstar:]         # (nDim, nDim, Tstar)

    return (
        Ystar,
        H0,
        H1,
        RR,
        F0_t,
        F1,
        Q_t,
        A_select,
        Ystar_m,
        Ystar_q,
    )
