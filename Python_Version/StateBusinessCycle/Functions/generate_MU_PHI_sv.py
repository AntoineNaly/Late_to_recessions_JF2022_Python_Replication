# -*- coding: utf-8 -*-

"""
generate_MU_PHI_sv.py
---------------------
Draws the common growth factor parameters via sequential Gibbs steps:
    1. φ_z   — AR(1) persistence of z_t      (Normal posterior, GLS)
    2. μ_z   — [μ_0, μ_1]                    (Normal posterior, GLS, rejection-sampled)
    3. σ²_{z,0}                               — always hardcoded to 1 (normalisation)
    4. h_z                                    — always hardcoded to 0 (equal vol across regimes)

The common growth factor z_t follows eq. (14):
    z_{t+1} = μ_z(S_{t+1}) + φ_z(S_{t+1})(z_t - μ_z(S_t)) + σ_z(S_{t+1})ε_{z,t+1}

where the regime-dependent mean and variance are (see footnote 18):
    μ_z(S_t) = μ_0 + μ_1 * S_t              (S_t=0: recession, S_t=1: expansion)
    σ²_z(S_t) = σ²_{z,0} * (1 + h_z * S_t)

Rearranging eq. (14) into a linear regression for GLS estimation:
    [z_{t+1} - μ_z(S_{t+1})] = φ_z [z_t - μ_z(S_t)] + σ_z(S_{t+1}) ε_{z,t+1}
or equivalently:
    Ystar_t = φ_z Xstar_t + σ_z(S_{t+1}) ε_{z,t+1}
where Ystar_t and Xstar_t are demeaned z values (GLS-scaled by σ_z(S_t)).

Rejection constraint for μ_1 (footnote 18): μ_1 > 0, ensuring that
the expansion mean exceeds the recession mean (μ_z(1) > μ_z(0)).

Normalisation (σ²_{z,0} = 1, h_z = 0): both parameters are computed
within the Gibbs step but immediately overridden to their fixed values.
σ²_{z,0} = 1 identifies the scale of the latent factor z_t; h_z = 0
imposes equal volatility across regimes.

Implementation notes
--------------------
The Inverse-Chi-Squared draw for σ² uses the identity:
    d / Σ u_k²   ~   IG(nn/2, d/2),   u_k ~ N(0,1)
which follows from Σ u_k² ~ χ²(nn) and the relationship between
the chi-squared and inverse-gamma distributions.

For [μ_0, μ_1], the raw intercept μ_0 is estimated in its transformed
form (from the GLS regression) and then back-transformed:
    μ_0 = μ_0_raw / (1 - φ_z)
"""

import numpy as np


def generate_mu_phi_sv(
    x_t,
    STT,
    param_macro_MH,
    R0_,
    T0_,
    R0_M,
    T0_M,
    D0_,
    V0_,
    rng,
):
    """
    Parameters
    ----------
    x_t           : (T,) ndarray  – current draw of common growth factor z_t
    STT           : (T,) ndarray  – regime path S_t (0=expansion, 1=recession)
                                    caller passes STT[2:] from main loop
    param_macro_MH : dict
        'paramMU'     : (2,) array [μ_0, μ_1]
        'Sigma2_0_cc' : float      σ²_{z,0}
        'h_cc'        : float      h_z
        'phi_cc'      : float      φ_z
    R0_   : (1,1) ndarray  – prior precision for φ_z  (eye(1)/4)
    T0_   : scalar or (1,) – prior mean for φ_z        (0)
    R0_M  : (2,2) ndarray  – prior precision for [μ_0_raw, μ_1]  (eye(2)/2)
    T0_M  : (2,) array     – prior mean for [μ_0_raw, μ_1]       ([0,0])
    D0_   : float           – prior scale for variance draw        (0)
    V0_   : float           – prior d.o.f. for variance draw       (0)
    rng   : np.random.Generator

    Returns
    -------
    phi_cc       : float  – updated φ_z
    paramMU      : (2,) ndarray – [μ_0, μ_1]
    Sigma2_0_cc  : float  – always 1 (normalised)
    h_cc         : float  – always 0 (equal vol across regimes)
    """

    # ------------------------------------------------------------------
    # Unpack current parameter values
    # ------------------------------------------------------------------
    x_t  = np.asarray(x_t,  dtype=float).reshape(-1)
    STT  = np.asarray(STT,  dtype=float).reshape(-1)
    T    = len(x_t)

    paramMU     = np.asarray(param_macro_MH["paramMU"], dtype=float).reshape(-1)
    mu_0        = paramMU[0]
    mu_1        = paramMU[1]
    Sigma2_0_cc = float(param_macro_MH["Sigma2_0_cc"])
    h_cc        = float(param_macro_MH["h_cc"])

    # Coerce prior arrays
    R0_  = np.asarray(R0_,  dtype=float).reshape(1, 1)
    T0_  = np.asarray(T0_,  dtype=float).reshape(-1)
    R0_M = np.asarray(R0_M, dtype=float).reshape(2, 2)
    T0_M = np.asarray(T0_M, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Step 1 – Draw φ_z
    # ------------------------------------------------------------------
    # From eq. (14), demeaning and rearranging:
    #   (z_{t+1} - μ_z(S_{t+1})) = φ_z (z_t - μ_z(S_t)) + σ_z(S_{t+1}) ε
    # GLS scaling by σ_z(S_{t+1}) = sqrt(σ²_{z,0}*(1+h_z*S_{t+1})) yields:
    #   Ystar_t / σ_t = (φ_z * Xstar_t) / σ_t + ε
    mu_t     = mu_0 + mu_1 * STT                        # (T,) μ_z(S_t) = μ_0 + μ_1*S_t
    sigma2_t = Sigma2_0_cc * (1.0 + h_cc * STT)         # (T,) σ²_z(S_t)

    Ystar_full = x_t - mu_t                              # (T,) z_t - μ_z(S_t)

    # GLS regressor/regressand for φ_z (aligned to period t+1 observations)
    Xstar_phi = Ystar_full[3:-1].reshape(-1, 1)         # (T-4, 1): z_{t-1} - μ_z(S_{t-1})
    Ystar_phi = Ystar_full[4:].reshape(-1, 1)            # (T-4, 1): z_t - μ_z(S_t)
    sigma_t   = np.sqrt(sigma2_t[4:]).reshape(-1, 1)    # (T-4, 1): σ_z(S_t) for GLS scaling

    # GLS: divide by conditional std dev σ_z(S_t)
    Xstar_phi = Xstar_phi / sigma_t
    Ystar_phi = Ystar_phi / sigma_t

    # Normal posterior for φ_z:
    # V_φ = (R0_ + Xstar'Xstar)^{-1},  φ_post = V_φ (R0_ T0_ + Xstar'Ystar)
    V_phi   = np.linalg.inv(R0_ + Xstar_phi.T @ Xstar_phi)  # (1,1)
    PHI_post = V_phi @ (R0_ @ T0_.reshape(-1, 1) + Xstar_phi.T @ Ystar_phi)  # (1,1)

    # Cholesky draw: φ_z ~ N(φ_post, V_φ)
    L_phi   = np.linalg.cholesky(V_phi)                 # (1,1) lower triangular
    PHI_G   = float(PHI_post.item()) + float(L_phi.item()) * rng.standard_normal()
    phi_cc  = PHI_G

    # ------------------------------------------------------------------
    # Step 2 – Draw [μ_0, μ_1]
    # ------------------------------------------------------------------
    # Rearranging eq. (14) for the mean parameters:
    #   z_{t+1} - φ_z z_t = μ_z(S_{t+1}) - φ_z μ_z(S_t) + σ_z(S_{t+1}) ε
    # GLS regression with design matrix [1, S_{t+1} - φ_z S_t]:
    Ystar_mu = (x_t[4:] - PHI_G * x_t[3:-1]).reshape(-1, 1)            # (T-4, 1)
    Xstar_mu = np.column_stack([
        np.ones(T - 4),
        STT[4:] - PHI_G * STT[3:-1],
    ])                                                                    # (T-4, 2)

    # GLS scaling by σ_z(S_t)
    Ystar_mu = Ystar_mu / sigma_t
    Xstar_mu = Xstar_mu / sigma_t

    # Normal posterior for [μ_0_raw, μ_1]:
    V_mu   = np.linalg.inv(R0_M + Xstar_mu.T @ Xstar_mu)                # (2,2)
    MU_post = V_mu @ (R0_M @ T0_M.reshape(-1, 1) + Xstar_mu.T @ Ystar_mu)  # (2,1)

    L_mu = np.linalg.cholesky(V_mu)                                       # (2,2) lower

    # Rejection sampling: draw until μ_1 > 0 (footnote 18).
    # μ_1 > 0 ensures the expansion mean exceeds the recession mean: μ_z(1) = μ_0 + μ_1 > μ_0.
    accept = False
    while not accept:
        MU_G = MU_post.reshape(-1) + L_mu @ rng.standard_normal(2)      # (2,)
        if MU_G[1] > 0:
            accept = True

    # Back-transform: the GLS regression estimates μ_0_raw = μ_0 * (1 - φ_z),
    # so the level-form intercept is μ_0 = μ_0_raw / (1 - φ_z).
    MU_G[0] = MU_G[0] / (1.0 - PHI_G)

    paramMU = MU_G                                                        # [μ_0, μ_1]
    mu_0    = paramMU[0]
    mu_1    = paramMU[1]

    # ------------------------------------------------------------------
    # Step 3 – Draw σ²_{z,0}
    # Computed from the IG posterior but immediately overridden to 1.
    # σ²_{z,0} = 1 is the normalisation restriction that identifies the
    # scale of the latent factor z_t.
    # ------------------------------------------------------------------
    mu_t    = mu_0 + mu_1 * STT
    Ystar_s = x_t - mu_t                                                 # (T,)

    e_mat_s = Ystar_s[4:] - PHI_G * Ystar_s[3:-1]                       # (T-4,)
    # GLS: divide by sqrt(1 + h_z * S_t) — regime-dependent denominator
    tempDenom_s = np.sqrt(1.0 + h_cc * STT[4:])
    e_mat_s = e_mat_s / tempDenom_s

    Tstar_s = len(e_mat_s)
    nn_s    = int(Tstar_s + V0_)
    d_s     = float(D0_) + float(e_mat_s @ e_mat_s)

    # IG draw: d / Σ u_k² ~ IG(nn_s/2, d_s/2)
    draws_s      = rng.standard_normal(nn_s)
    Sigma2_0_cc  = d_s / float(draws_s @ draws_s)  # noqa: F841  (overridden below)

    # ------------------------------------------------------------------
    # Step 4 – Draw h_z
    # Computed from the IG posterior but immediately overridden to 0.
    # h_z = 0 imposes equal volatility across regimes.
    # ------------------------------------------------------------------
    e_mat_h = Ystar_s[4:] - PHI_G * Ystar_s[3:-1]                       # (T-4,)
    # Scale by σ²_{z,0} from just-drawn value before rejection
    e_mat_h = e_mat_h / np.sqrt(Sigma2_0_cc)
    # Keep only recession periods (h_z captures the recession volatility premium)
    rec_mask = STT[4:].astype(bool)
    e_mat_h  = e_mat_h[rec_mask]

    Tstar_h = len(e_mat_h)
    nn_h    = int(Tstar_h + V0_)
    d_h     = float(D0_) + float(e_mat_h @ e_mat_h)

    # Rejection sampling: draw until h_hat > 2/3 (ensures σ²_z(S_t) > 0 for all S_t)
    accept = False
    while not accept:
        draws_h = rng.standard_normal(nn_h)
        h_hat   = d_h / float(draws_h @ draws_h)
        if h_hat > 2.0 / 3.0:
            accept = True

    h_cc = h_hat - 1.0  # noqa: F841  (overridden below)

    # ------------------------------------------------------------------
    # Normalisation: enforce σ²_{z,0} = 1 and h_z = 0 (equal volatility).
    # These values override the draws above to maintain identification.
    # ------------------------------------------------------------------
    h_cc        = 0.0
    Sigma2_0_cc = 1.0

    return phi_cc, paramMU, Sigma2_0_cc, h_cc
