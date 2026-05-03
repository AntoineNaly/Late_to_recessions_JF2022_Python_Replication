# -*- coding: utf-8 -*-

"""
generate_gamma_macro.py
-----------------------
Draws the factor loading γ_i for one macro variable from its Gaussian
conjugate posterior.

The factor loading γ_i relates the observable y_{i,t} to the common growth
factor z_t via eq. (13):
    y_{i,t+1} = γ_i z_{t+1} + e_{i,t+1}

After pre-whitening to remove the AR(1) in e_{i,t} (eq. 15), the regression
used to draw γ_i becomes (from eq. IA.15):
    y*_{i,t+1} = γ_i z_{t+1} - γ_i ψ_i z_t + σ_i ε_{i,t+1}
               = γ_i Xstar_t + σ_i ε_{i,t+1}

where Ystar_t = y_{i,t+1} - ψ_i y_{i,t}  and  Xstar_t = z_{t+1} - ψ_i z_t.

Two cases controlled by (selectVar, nVars):
    selectVar != nVars  ->  scalar γ_i, pre-whitened regressors (as above)
    selectVar == nVars  ->  4-dimensional γ_i for the last monthly variable,
                            which uses 4 lags of z_t in its measurement equation.
                            (This case is never triggered for quarterly variables.)

Posterior draw (Normal conjugate):
    Prior:      γ_i ~ N(T0, R0^{-1})
    Likelihood: Ystar = Xstar * γ_i + σ_i u,   u ~ N(0, I)
    Posterior covariance: V_γ = (R0 + σ_i^{-2} Xstar'Xstar)^{-1}
    Posterior mean:       γ_post = V_γ (R0 T0 + σ_i^{-2} Xstar'Ystar)
    Draw:                 γ_i ~ N(γ_post, V_γ)  via Cholesky decomposition
"""

import numpy as np


def generate_gamma_macro(
    y,
    x_t,
    PSI_i,
    SIG2_i,
    R00_,
    T00_,
    R00_4,
    T00_4,
    selectVar,
    nVars,
    rng,
):
    """
    Parameters
    ----------
    y        : (T_obs,) ndarray  -- NaN-free observations
    x_t      : (T_obs,) ndarray  -- NaN-aligned common factor z_t
    PSI_i    : float             -- current ψ_i draw for this variable
    SIG2_i   : float             -- current σ²_{e,i} draw
    R00_     : float or (1,1)    -- prior precision, scalar case  (1/4)
    T00_     : float or (1,)     -- prior mean, scalar case       (0)
    R00_4    : (4,4) ndarray     -- prior precision, 4-lag case   (eye(4))
    T00_4    : (4,) ndarray      -- prior mean, 4-lag case        ([0,0,0,0])
    selectVar : int              -- 1-indexed variable index
    nVars     : int
    rng       : np.random.Generator

    Returns
    -------
    gamma_macro : (k,) ndarray   k=1 for non-last, k=4 for last monthly variable
    """

    y    = np.asarray(y,   dtype=float).reshape(-1)
    x_t  = np.asarray(x_t, dtype=float).reshape(-1)
    psi_1 = float(np.asarray(PSI_i).flat[0])

    # ------------------------------------------------------------------
    # Pre-whiten observations and regressors (from eq. IA.15):
    #   Ystar_t = y_{i,t+1} - ψ_i y_{i,t}
    #   Xstar_t = z_{t+1}   - ψ_i z_t
    # This removes the AR(1) correlation in e_{i,t}, yielding a regression
    # with i.i.d. errors that admits a clean Gaussian conjugate posterior.
    # ------------------------------------------------------------------
    if selectVar != nVars:
        # Non-last variable: scalar γ_i
        Ystar = y[1:]  - psi_1 * y[:-1]      # y*_{i,t+1} = y_{i,t+1} - ψ_i y_{i,t}
        Xstar = x_t[1:] - psi_1 * x_t[:-1]  # x*_{t+1}   = z_{t+1}   - ψ_i z_t
        Xstar = Xstar.reshape(-1, 1)           # (T_obs-1, 1)
        Ystar = Ystar.reshape(-1, 1)           # (T_obs-1, 1)

        R0 = float(np.asarray(R00_).flat[0])  # scalar prior precision
        T0 = np.array([[float(np.asarray(T00_).flat[0])]])  # (1,1)
        k  = 1

    else:
        # Last monthly variable (weekly jobless claims aggregated to monthly):
        # uses 4 lags of z_t, so the pre-whitened regression has 4 regressors.
        # Ystar*_t = y_{i,t+8} - ψ_i y_{i,t+7}
        # Xstar_k  = z_{t+8-k} - ψ_i z_{t+7-k}  for k=1,2,3,4
        Ystar   = (y[7:]    - psi_1 * y[6:-1]).reshape(-1, 1)
        XSTAR_1 = x_t[7:]    - psi_1 * x_t[6:-1]
        XSTAR_2 = x_t[6:-1]  - psi_1 * x_t[5:-2]
        XSTAR_3 = x_t[5:-2]  - psi_1 * x_t[4:-3]
        XSTAR_4 = x_t[4:-3]  - psi_1 * x_t[3:-4]
        Xstar   = np.column_stack([XSTAR_1, XSTAR_2, XSTAR_3, XSTAR_4])  # (T_obs-7, 4)

        R0 = np.asarray(R00_4, dtype=float).reshape(4, 4)
        T0 = np.asarray(T00_4, dtype=float).reshape(4, 1)
        k  = 4

    sig2_inv = 1.0 / float(SIG2_i)

    # ------------------------------------------------------------------
    # Gaussian posterior for γ_i:
    #
    # Prior:      γ_i ~ N(T0, R0^{-1})
    #   → prior density ∝ exp(-½ (γ_i - T0)' R0 (γ_i - T0))
    # Likelihood: Ystar = Xstar γ_i + σ_i u,  u ~ N(0,I)
    #   → ∝ exp(-½σ_i^{-2} (Ystar - Xstar γ_i)' (Ystar - Xstar γ_i))
    #
    # Posterior (multiply prior × likelihood and complete the square):
    #   V_γ    = (R0 + σ_i^{-2} Xstar'Xstar)^{-1}   (posterior covariance)
    #   γ_post = V_γ (R0 T0 + σ_i^{-2} Xstar'Ystar)  (posterior mean)
    #
    # The posterior mean is a precision-weighted average:
    #   R0 T0 = prior pulling toward T0=0 with strength R0
    #   σ_i^{-2} Xstar'Ystar = OLS numerator scaled by data precision
    # ------------------------------------------------------------------
    if k == 1:
        r0_scalar = float(R0)
        t0_scalar = float(T0.flat[0])
        V_g    = 1.0 / (r0_scalar + sig2_inv * float((Xstar.T @ Xstar).item()))
        gam    = V_g * (r0_scalar * t0_scalar + sig2_inv * float((Xstar.T @ Ystar).item()))
        draw   = np.array([gam + np.sqrt(V_g) * rng.standard_normal()])
    else:
        V_g  = np.linalg.inv(R0 + sig2_inv * (Xstar.T @ Xstar))   # (4,4)
        V_g  = 0.5 * (V_g + V_g.T)                                  # symmetrise for Cholesky
        gam  = V_g @ (R0 @ T0 + sig2_inv * (Xstar.T @ Ystar))     # (4,1)
        L    = np.linalg.cholesky(V_g)                             # lower Cholesky: V_g = L L'
        draw = gam.flatten() + L @ rng.standard_normal(k)          # γ ~ N(γ_post, V_γ)

    return np.asarray(draw, dtype=float).reshape(-1)
