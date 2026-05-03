# -*- coding: utf-8 -*-

"""
specifyPriorsGibbsMacro.py
--------------------------
Defines prior hyperparameters for all model parameters in the macroeconomic
block.  The priors listed here correspond to the "fairly agnostic" distributions
reported in Table IV of the paper.  Distributions used are N (normal), U (uniform),
and IG (inverse-gamma).

Parameters and their roles in the model equations:

Transition probabilities (p, q) — Markov chain for S_t in eq. (14):
    Beta(1, 1) priors for p = P(stay in recession) and q = P(stay in expansion).
    Beta(1,1) = Uniform[0,1]: no prior information on persistence.

AR(1) coefficient ψ_i — idiosyncratic process eq. (15):
    Prior: ψ_i ~ N(T0_V, R0_V^{-1}) = N(0, 4*I)
    R0_V = eye(1)/4 is the prior precision; T0_V = 0 is the prior mean.

Idiosyncratic variance σ²_{e,i} — eq. (15):
    Prior: σ²_{e,i} ~ IG(V0_/2, D0_/2) with V0_ = D0_ = 0 (diffuse/improper).

Factor loading γ_i — eq. (13), scalar case (all but last monthly variable):
    Prior: γ_i ~ N(T00_, R00_^{-1}) = N(0, 4)

Factor loading γ_i — eq. (13), 4-lag case (last monthly variable):
    Prior: γ_i ~ N(T00_4, R00_4^{-1}) = N(0, I_4)

AR(1) persistence φ_z — common growth factor eq. (14):
    Prior: φ_z ~ N(T0_, R0_^{-1}) = N(0, 4*I)

Regime-dependent means [μ_0, μ_1] — eq. (14), μ_z(S_t) = μ_0 + μ_1*S_t:
    Prior: [μ_0, μ_1] ~ N(T0_M, R0_M^{-1}) = N([0,0], 2*I_2)
    Rejection sampling enforces μ_1 > 0 (expansion mean exceeds recession mean).
"""

import numpy as np


def specify_priors_gibbs_macro():
    """
    Build the priorsMacroGibbs structure and return scalar hyperparameters
    needed in initialization and sampling.

    Returns
    -------
    priorsMacroGibbs : dict
        Dictionary of prior hyperparameters:
            R0_V, T0_V      -- for ψ_i in eq. (15)
            V0_, D0_        -- for σ²_{e,i} in eq. (15)
            R00_, T00_      -- for scalar γ_i in eq. (13)
            R00_4, T00_4    -- for 4-lag γ_i in eq. (13)
            R0_, T0_        -- for φ_z in eq. (14)
            R0_M, T0_M      -- for [μ_0, μ_1] in eq. (14)
    markov_priors : dict
        Beta prior pseudo-counts for Markov transition probabilities p, q:
            U1_01_, U1_00_, U1_10_, U1_11_
    """
    # Beta(1,1) = Uniform[0,1] priors on Markov transition probabilities p, q
    markov_priors = {
        "U1_01_": 1.0,
        "U1_00_": 1.0,
        "U1_10_": 1.0,
        "U1_11_": 1.0,
    }

    # Prior for ψ_i (AR(1) coefficient of idiosyncratic component, eq. 15):
    # ψ_i ~ N(0, 4)  →  precision R0_V = 1/4, mean T0_V = 0
    R0_V = np.eye(1) / 4.0
    T0_V = np.array([0.0])

    # Prior for σ²_{e,i} (idiosyncratic innovation variance, eq. 15):
    # Diffuse IG prior with V0_ = D0_ = 0 (improper; dominated by likelihood)
    D0_ = 0.0
    V0_ = 0.0

    # Prior for scalar γ_i (factor loading for general variables, eq. 13):
    # γ_i ~ N(0, 4)  →  precision R00_ = 1/4, mean T00_ = 0
    R00_ = np.array([[1.0 / 4.0]])
    T00_ = np.array([0.0])

    # Prior for 4-lag γ_i (last monthly variable with weekly-to-monthly aggregation, eq. 13):
    # γ_i ~ N(0, I_4)  →  precision R00_4 = I_4, mean T00_4 = 0
    R00_4 = np.eye(4)
    T00_4 = np.zeros(4)

    # Prior for φ_z (AR(1) persistence of the common growth factor, eq. 14):
    # φ_z ~ N(0, 4)  →  precision R0_ = 1/4, mean T0_ = 0
    R0_ = np.eye(1) / 4.0
    T0_ = np.array([0.0])

    # Prior for [μ_0, μ_1] (regime-dependent mean of z_t, eq. 14):
    # [μ_0, μ_1] ~ N([0,0], 2*I_2)  →  precision R0_M = I_2/2, mean T0_M = [0,0]
    R0_M = np.eye(2) / 2.0
    T0_M = np.array([0.0, 0.0])

    priorsMacroGibbs = {
        "R0_V": R0_V,
        "T0_V": T0_V,
        "V0_": V0_,
        "D0_": D0_,
        "R00_": R00_,
        "T00_": T00_,
        "R00_4": R00_4,
        "T00_4": T00_4,
        "R0_": R0_,
        "T0_": T0_,
        "R0_M": R0_M,
        "T0_M": T0_M,
    }

    return priorsMacroGibbs, markov_priors
