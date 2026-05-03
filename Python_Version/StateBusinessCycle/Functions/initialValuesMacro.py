# -*- coding: utf-8 -*-

"""
initialValuesMacro.py
---------------------
Computes OLS-based initial values for all Gibbs sampler parameters using
the observed data.  These starting values are used only to initialise the
chain; the Gibbs sampler then iterates to the posterior.

What is computed
----------------
The model equations being initialised are:

    y_{i,t+1} = γ_i z_{t+1} + e_{i,t+1}                           (eq. 13)
    e_{i,t+1} = μ_i + ψ_i e_{i,t} + σ_i ε_{i,t+1}                (eq. 15)
    z_{t+1} = μ_z(S_{t+1}) + φ_z(z_t - μ_z(S_t)) + σ_z ε_{z,t+1} (eq. 14)

Monthly variables (first N_m-1 use scalar γ; last uses 4-lag γ):
    gamma_macro_m : (N_m+3,) — factor loadings from OLS of eq. (13)
    psi_macro_m   : (N_m,)   — AR(1) coefficients from OLS of eq. (15) residuals
    SIG2_i_macro_m: (N_m,)   — innovation variances from OLS of eq. (15) residuals

Quarterly variables (all scalar γ):
    gamma_macro_q : (N_q,)
    psi_macro_q   : (N_q,)
    SIG2_i_macro_q: (N_q,)

Common growth factor (eq. 14):
    phi_cc       : float  — AR(1) persistence, estimated by OLS on the factor proxy
    paramMU      : (2,)   — [μ_0, μ_1] = [-2, 2.5], hardcoded (Table IV prior centre)
    Sigma2_0_cc  : 1.0    — hardcoded normalisation (identified by this restriction)
    h_cc         : -0.3   — hardcoded initial value (overwritten to 0 after first MH step)

Markov transition parameters (transition matrix for eq. 14's regime S_t):
    paramProb : (2,) — [A1TT, B1TT] drawn from Beta posteriors using NBER data

After parameter estimation, columns with >99% NaN are dropped from
yy_monthly (and corresponding entries removed), and columns with >80%
NaN are dropped from yy_quarterly.

MATLAB design notes
-------------------
- No-intercept OLS (fitlm 'intercept' false) matches np.linalg.lstsq.
- std() uses ddof=1 denominator (N-1).
- paramMU = [-2; 2.5] is hardcoded — not estimated from data.
- h_cc = -0.3 is the hardcoded initial value; generate_mu_phi_sv always
  overwrites h_cc=0 after the first Kalman pass.
- gamma_macro_m has length N_m+3:
    positions 0..N_m-2 : scalar loadings for first N_m-1 monthly vars
    positions N_m-1..N_m+2 : four-lag loadings for the last monthly var
- After NaN-column removal for monthly, the indexing mask is
  [~perNaN_monthly, True, True, True] so the last 3 extra-lag entries
  are always kept.
"""

import numpy as np

try:
    from .generate_ChangeState import generate_change_state
except ImportError:
    from generate_ChangeState import generate_change_state


def initial_values_macro(
    yy_monthly,
    yy_quarterly,
    NBER_rec_index,
    markov_priors,
    rng,
):
    """
    Parameters
    ----------
    yy_monthly   : (T, N_m) ndarray  – standardised monthly data
    yy_quarterly : (T, N_q) ndarray  – standardised quarterly data
    NBER_rec_index : (T,) array       – 1 = recession, 0 = expansion
    markov_priors : dict
        Keys: 'U1_01_', 'U1_00_', 'U1_10_'  (Beta prior pseudo-counts)
    rng : np.random.Generator

    Returns
    -------
    param_macro_MH   : dict
    param_macro_gibbs : dict
    s_t              : (T,) ndarray  — initial regime path (0=recession,1=expansion)
    yy_monthly       : (T, N_m_clean) ndarray  — after NaN-column removal
    yy_quarterly     : (T, N_q_clean) ndarray  — after NaN-column removal
    N_m              : int  — updated column count
    N_q              : int  — updated column count
    """

    yy_monthly   = np.array(yy_monthly,   dtype=float)
    yy_quarterly = np.array(yy_quarterly, dtype=float)
    nber         = np.asarray(NBER_rec_index, dtype=float).reshape(-1)

    T,   N_m = yy_monthly.shape
    _,   N_q = yy_quarterly.shape

    # ------------------------------------------------------------------
    # Initial regime path S_t from NBER recession dates:
    #   NBER=1 → recession → S_t=0
    #   NBER=0 → expansion → S_t=1
    # This is the starting value for the Hamilton filter (eq. 16 / Section C.1).
    # ------------------------------------------------------------------
    s_t = (~nber.astype(bool)).astype(float)   # (T,)

    # ------------------------------------------------------------------
    # Common factor proxy: cross-sectional mean of monthly variables,
    # standardised.  Used as a stand-in for z_t to initialise γ_i via
    # OLS of eq. (13): y_{i,t} = γ_i * z_t + e_{i,t}.
    # ------------------------------------------------------------------
    x_mean = np.nanmean(yy_monthly, axis=1)                 # (T,)
    x_t    = x_mean / np.std(x_mean, ddof=1)               # (T,)

    # ------------------------------------------------------------------
    # Monthly γ (factor loadings) — OLS of eq. (13) on common factor proxy
    # gamma_macro_m has length N_m+3:
    #   indices 0..N_m-2  : scalar γ for first N_m-1 variables
    #   indices N_m-1..N_m+2 : 4-lag γ for the last variable
    # ------------------------------------------------------------------
    gamma_macro_m = np.zeros(N_m + 3)
    e_t_m         = np.full((T, N_m), np.nan)

    # First N_m-1 variables: scalar OLS of y_{i,t} = γ_i * x_t (no intercept)
    for i in range(N_m - 1):
        y_i    = yy_monthly[:, i]
        mask_i = ~np.isnan(y_i)
        X_i    = x_t[mask_i].reshape(-1, 1)
        y_obs  = y_i[mask_i]
        gamma_i, *_ = np.linalg.lstsq(X_i, y_obs, rcond=None)
        gamma_macro_m[i] = float(np.asarray(gamma_i).item())
        resid = np.full(T, np.nan)
        resid[mask_i] = y_obs - X_i.flatten() * float(np.asarray(gamma_i).item())
        e_t_m[:, i] = resid

    # Last variable: 4-lag OLS, reflecting the weekly-to-monthly aggregation
    # in the measurement equation for this series.
    # Regressors: x_t(4:end-1), x_t(3:end-2), x_t(2:end-3), x_t(1:end-4)
    Xaux = np.column_stack([
        x_t[3:-1],    # x_t(4:end-1)
        x_t[2:-2],    # x_t(3:end-2)
        x_t[1:-3],    # x_t(2:end-3)
        x_t[:-4],     # x_t(1:end-4)
    ])                                               # (T-4, 4)
    Yaux = yy_monthly[4:, -1]                        # (T-4,)

    # Handle NaNs in Yaux
    mask_last        = ~np.isnan(Yaux)
    gamma_last, *_   = np.linalg.lstsq(Xaux[mask_last], Yaux[mask_last], rcond=None)
    gamma_macro_m[N_m - 1 : N_m + 3] = gamma_last   # 4 values

    resid_last = np.full(T, np.nan)
    resid_last[4:][mask_last] = Yaux[mask_last] - Xaux[mask_last] @ gamma_last
    e_t_m[:, -1] = resid_last

    # ------------------------------------------------------------------
    # Monthly ψ and σ²_{e,i} — OLS of eq. (15): e_{i,t+1} = ψ_i * e_{i,t} + residual
    # ------------------------------------------------------------------
    psi_macro_m    = np.zeros(N_m)
    SIG2_i_macro_m = np.zeros(N_m)

    for i in range(N_m):
        e_i   = e_t_m[:, i]
        valid = ~np.isnan(e_i)
        e_sel = e_i[valid]

        X_ar  = e_sel[:-1].reshape(-1, 1)
        y_ar  = e_sel[1:]
        psi_i, *_ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
        psi_macro_m[i] = float(np.asarray(psi_i).item())

        resid_ar = y_ar - X_ar.flatten() * float(np.asarray(psi_i).item())
        SIG2_i_macro_m[i] = np.nanmean(resid_ar ** 2)

    # ------------------------------------------------------------------
    # Quarterly γ — scalar OLS of eq. (13) for each quarterly variable
    # ------------------------------------------------------------------
    gamma_macro_q  = np.zeros(N_q)
    e_t_q          = np.full((T, N_q), np.nan)

    for i in range(N_q):
        y_i    = yy_quarterly[:, i]
        mask_i = ~np.isnan(y_i)
        X_i    = x_t[mask_i].reshape(-1, 1)
        y_obs  = y_i[mask_i]
        gamma_i, *_ = np.linalg.lstsq(X_i, y_obs, rcond=None)
        gamma_macro_q[i] = float(np.asarray(gamma_i).item())
        resid = np.full(T, np.nan)
        resid[mask_i] = y_obs - X_i.flatten() * float(np.asarray(gamma_i).item())
        e_t_q[:, i] = resid

    # ------------------------------------------------------------------
    # Quarterly ψ and σ²_{e,i} — OLS of eq. (15) on quarterly residuals
    # ------------------------------------------------------------------
    psi_macro_q    = np.zeros(N_q)
    SIG2_i_macro_q = np.zeros(N_q)

    for i in range(N_q):
        e_i   = e_t_q[:, i]
        valid = ~np.isnan(e_i)
        e_sel = e_i[valid]

        X_ar  = e_sel[:-1].reshape(-1, 1)
        y_ar  = e_sel[1:]
        psi_i, *_ = np.linalg.lstsq(X_ar, y_ar, rcond=None)
        psi_macro_q[i] = float(np.asarray(psi_i).item())

        resid_ar = y_ar - X_ar.flatten() * float(np.asarray(psi_i).item())
        SIG2_i_macro_q[i] = np.nanmean(resid_ar ** 2)

    # ------------------------------------------------------------------
    # Common factor φ_z — OLS of eq. (14) on the factor proxy:
    # x_t = φ_z * x_{t-1} + residual (no intercept; proxy already demeaned)
    # ------------------------------------------------------------------
    X_phi    = x_t[:-1].reshape(-1, 1)
    y_phi    = x_t[1:]
    phi_hat, *_ = np.linalg.lstsq(X_phi, y_phi, rcond=None)
    phi_cc   = float(np.asarray(phi_hat).item())

    # Hardcoded initial values from eq. (14) — not estimated from data.
    # μ_0 = -2, μ_1 = 2.5 are centred near the prior means in Table IV.
    # σ²_{z,0} = 1 is the normalisation restriction that identifies the scale of z_t.
    # h_z = -0.3 is overwritten to 0 by generate_mu_phi_sv after the first Kalman pass.
    paramMU     = np.array([-2.0, 2.5])
    Sigma2_0_cc = 1.0
    h_cc        = -0.3

    # ------------------------------------------------------------------
    # Markov transition probabilities p, q from eq. (14):
    #   p = P(stay in recession | currently recession)
    #   q = P(stay in expansion | currently expansion)
    # Posterior is Beta given transition counts from the NBER-based s_t.
    # Beta(α, β) is the conjugate prior to the Binomial likelihood;
    # α/(α+β) = prior mean, and α+β controls prior strength.
    # ------------------------------------------------------------------
    U1_01_ = float(markov_priors["U1_01_"])
    U1_00_ = float(markov_priors["U1_00_"])
    U1_10_ = float(markov_priors["U1_10_"])

    states_for_tranmat = (s_t[4:] + 1).astype(int)   # states in {1,2}
    tranmat = generate_change_state(states_for_tranmat, states=[1, 2])
    # tranmat[0,0]=count(0→0), tranmat[0,1]=count(0→1)
    # tranmat[1,0]=count(1→0), tranmat[1,1]=count(1→1)

    A1TT = rng.beta(tranmat[0, 1] + U1_01_,
                    tranmat[0, 0] + U1_00_)   # P(0→1) = 1-p, probability of leaving recession
    B1TT = rng.beta(tranmat[1, 0] + U1_10_,
                    tranmat[1, 1] + U1_10_)   # P(1→0) = 1-q, probability of leaving expansion

    paramProb = np.array([A1TT, B1TT])

    # ------------------------------------------------------------------
    # Pack MH-level parameters (φ_z, μ_z, σ²_{z,0}, h_z, p, q)
    # ------------------------------------------------------------------
    param_macro_MH = dict(
        paramMU      = paramMU,
        Sigma2_0_cc  = Sigma2_0_cc,
        h_cc         = h_cc,
        phi_cc       = phi_cc,
        paramProb    = paramProb,
    )

    # ------------------------------------------------------------------
    # Pack Gibbs-level parameters (γ_i, ψ_i, σ²_{e,i} for all series)
    # ------------------------------------------------------------------
    param_macro_gibbs = dict(
        gamma_macro_m   = gamma_macro_m,
        psi_macro_m     = psi_macro_m,
        SIG2_i_macro_m  = SIG2_i_macro_m,
        gamma_macro_q   = gamma_macro_q,
        psi_macro_q     = psi_macro_q,
        SIG2_i_macro_q  = SIG2_i_macro_q,
    )

    # ------------------------------------------------------------------
    # NaN-column removal — monthly (>99% NaN threshold)
    # Variables that are almost entirely missing are dropped; their
    # parameter entries are removed from the corresponding arrays.
    # ------------------------------------------------------------------
    frac_nan_m    = np.isnan(yy_monthly).sum(axis=0) / T    # (N_m,)
    keep_m        = frac_nan_m <= 0.99                       # (N_m,) bool

    yy_monthly            = yy_monthly[:, keep_m]
    SIG2_i_macro_m        = SIG2_i_macro_m[keep_m]
    psi_macro_m           = psi_macro_m[keep_m]

    # For gamma_macro_m: keep surviving scalar entries + always keep the
    # last 3 extra-lag entries (which belong to the last variable's 4-lag γ)
    index_gamma_m         = np.concatenate([keep_m, [True, True, True]])
    gamma_macro_m         = gamma_macro_m[index_gamma_m]

    N_m = yy_monthly.shape[1]

    param_macro_gibbs["gamma_macro_m"]   = gamma_macro_m
    param_macro_gibbs["psi_macro_m"]     = psi_macro_m
    param_macro_gibbs["SIG2_i_macro_m"]  = SIG2_i_macro_m

    # ------------------------------------------------------------------
    # NaN-column removal — quarterly (>80% NaN threshold)
    # ------------------------------------------------------------------
    frac_nan_q    = np.isnan(yy_quarterly).sum(axis=0) / T  # (N_q,)
    keep_q        = frac_nan_q <= 0.80                       # (N_q,) bool

    yy_quarterly           = yy_quarterly[:, keep_q]
    SIG2_i_macro_q         = SIG2_i_macro_q[keep_q]
    psi_macro_q            = psi_macro_q[keep_q]
    gamma_macro_q          = gamma_macro_q[keep_q]

    N_q = yy_quarterly.shape[1]

    param_macro_gibbs["gamma_macro_q"]   = gamma_macro_q
    param_macro_gibbs["psi_macro_q"]     = psi_macro_q
    param_macro_gibbs["SIG2_i_macro_q"]  = SIG2_i_macro_q

    return (
        param_macro_MH,
        param_macro_gibbs,
        s_t,
        yy_monthly,
        yy_quarterly,
        N_m,
        N_q,
    )
