# -*- coding: utf-8 -*-

"""
generate_PSIandSIG_macro.py
---------------------------
Draws ψ_i (idiosyncratic AR(1) coefficient) and σ²_{e,i} (idiosyncratic
innovation variance) for one macro variable from their conjugate posteriors.

The idiosyncratic component e_{i,t} follows eq. (15):
    e_{i,t+1} = μ_i + ψ_i e_{i,t} + σ_i ε_{i,t+1},   ε_{i,t+1} ~ N(0,1)

Given the factor loading γ_i and the drawn z_t path, the reconstructed
idiosyncratic component is:
    ê_{i,t} = y_{i,t} - γ_i z_t      (from eq. 13, isolating e_{i,t})

Pre-whitening to obtain i.i.d. residuals for the regression:
    ê*_{i,t} = ê_{i,t} - ψ_i ê_{i,t-1}   (from eq. 15, one-step AR residual)

The scalar or 4-lag case is controlled by (selectVar, nVars):
    selectVar != nVars  ->  scalar γ, AR(1) pre-whitened regressors
    selectVar == nVars  ->  4-lag γ, 4-lag pre-whitened regressors
                            (never triggered for quarterly variables)

Posterior draws
---------------
ψ_i draw (Normal posterior):
    Prior: ψ_i ~ N(T0_V, R0_V^{-1})
    Likelihood: Ystar = Xstar * ψ_i + σ_i * u,  u ~ N(0,I)
    Posterior precision:  V_ψ^{-1} = R0_V + σ_i^{-2} * Xstar'Xstar
    Posterior mean:       ψ_post   = V_ψ * (R0_V*T0_V + σ_i^{-2} * Xstar'Ystar)
    Draw:                 ψ ~ N(ψ_post, V_ψ)

σ²_{e,i} draw (Inverse-Gamma posterior):
    Prior: σ²_{e,i} ~ IG(V0_/2, D0_/2)  (diffuse with V0_=D0_=0)
    Posterior: σ²_{e,i} | ψ_i, data ~ IG((V0_+T*)/2, (D0_+ε̂'ε̂)/2)
    Computed as: d / Σ u_k²  where u_k ~ N(0,1) and d = D0_ + ε̂'ε̂
"""

import numpy as np


def generate_psi_and_sig_macro(
    y,
    x_t,
    gamma_i,
    SIG2_i,
    R0_V,
    T0_V,
    V0_,
    D0_,
    selectVar,
    nVars,
    rng,
):
    """
    Parameters
    ----------
    y        : (T_obs,) ndarray  -- NaN-free observations for this variable
    x_t      : (T_obs,) ndarray  -- NaN-aligned common factor z_t
    gamma_i  : float or (4,) ndarray
        Scalar γ_i for non-last variables; length-4 for last monthly variable.
    SIG2_i   : float             -- current σ²_{e,i} draw
    R0_V     : float or (1,1)    -- prior precision for ψ_i  (1/4)
    T0_V     : float or (1,)     -- prior mean for ψ_i       (0)
    V0_      : float  -- prior degrees of freedom for σ²_{e,i}  (0)
    D0_      : float  -- prior scale for σ²_{e,i}               (0)
    selectVar : int   -- 1-indexed variable index
    nVars     : int   -- total number of variables in this call
    rng       : np.random.Generator

    Returns
    -------
    psi_macro : float
    SIG2_i    : float
    """

    y    = np.asarray(y,   dtype=float).reshape(-1)
    x_t  = np.asarray(x_t, dtype=float).reshape(-1)

    r0 = float(np.asarray(R0_V).flat[0])   # scalar prior precision for ψ_i
    t0 = float(np.asarray(T0_V).flat[0])   # scalar prior mean for ψ_i

    # ------------------------------------------------------------------
    # Build Ystar and Xstar from the reconstructed idiosyncratic component.
    # From eq. (13): ê_{i,t} = y_{i,t} - γ_i z_t.
    # From eq. (15): Ystar_t = ê_{i,t} - ψ_i ê_{i,t-1} (AR(1) regression residual).
    # Pre-whitening yields i.i.d. errors suitable for a conjugate Gaussian posterior.
    # ------------------------------------------------------------------
    if selectVar != nVars:
        # Non-last variable: scalar γ_i
        gc = float(np.asarray(gamma_i).flat[0])
        Ystar_full = y - gc * x_t                     # (T_obs,): ê_{i,t} = y_{i,t} - γ_i z_t
        Xstar = Ystar_full[:-1]                       # (T_obs-1,): ê_{i,t-1} (AR regressor)
        Ystar = Ystar_full[1:]                        # (T_obs-1,): ê_{i,t}   (AR regressand)

    else:
        # Last monthly variable: 4-lag γ vector.
        # ê_{i,t} = y_{i,t} - γ_0 z_t - γ_1 z_{t-1} - γ_2 z_{t-2} - γ_3 z_{t-3}
        g = np.asarray(gamma_i, dtype=float).reshape(-1)   # (4,)
        Ystar_full = (
            y[3:]
            - g[0] * x_t[3:]
            - g[1] * x_t[2:-1]
            - g[2] * x_t[1:-2]
            - g[3] * x_t[:-3]
        )                                                   # (T_obs-3,)
        Xstar = Ystar_full[3:-1]                            # (T_obs-7,): AR regressor
        Ystar = Ystar_full[4:]                              # (T_obs-8,): AR regressand

    Tstar   = len(Ystar)
    sig2_inv = 1.0 / float(SIG2_i)

    # ------------------------------------------------------------------
    # Draw ψ_i | σ²_{e,i}, data — Normal posterior
    #
    # Prior: ψ_i ~ N(t0, r0^{-1})
    # Likelihood: Ystar = Xstar * ψ_i + σ_i u,  u ~ N(0,I)
    #
    # Posterior precision = prior precision + data precision:
    #   V_ψ^{-1} = r0 + σ_i^{-2} * Σ ê²_{i,t-1}
    # Posterior mean = precision-weighted average of prior and OLS:
    #   ψ_post = V_ψ * (r0*t0 + σ_i^{-2} * Σ ê_{i,t-1} ê_{i,t})
    # ------------------------------------------------------------------
    V_psi    = 1.0 / (r0 + sig2_inv * float(Xstar @ Xstar))   # posterior variance
    PSI_post = V_psi * (r0 * t0 + sig2_inv * float(Xstar @ Ystar))  # posterior mean
    psi_draw = PSI_post + np.sqrt(V_psi) * rng.standard_normal()

    # ------------------------------------------------------------------
    # Draw σ²_{e,i} | ψ_i, data — Inverse-Gamma posterior
    #
    # Residuals from eq. (15) using just-drawn ψ_i:
    #   ε̂_{i,t} = ê_{i,t} - ψ_i ê_{i,t-1}
    #
    # IG posterior: σ²_{e,i} | ψ_i, data ~ IG((V0_+T*)/2, (D0_+ε̂'ε̂)/2)
    # Sampled as: d / Σ u_k²,  u_k ~ N(0,1),  d = D0_ + ε̂'ε̂
    # This is equivalent to d / χ²(nn) where χ²(nn) = Σ u_k².
    # ------------------------------------------------------------------
    e_mat  = Ystar - Xstar * psi_draw           # AR(1) residuals ε̂_{i,t}
    nn     = int(Tstar + V0_)                   # posterior degrees of freedom
    d      = float(D0_) + float(e_mat @ e_mat)  # posterior scale
    draws  = rng.standard_normal(max(nn, 1))    # nn independent N(0,1) draws
    SIG2_i = d / float(draws @ draws)           # d / χ²(nn) ~ IG(nn/2, d/2)

    return float(psi_draw), float(SIG2_i)
