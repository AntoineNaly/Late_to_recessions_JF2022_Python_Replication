# -*- coding: utf-8 -*-

"""
priors.py
----------
Translated from MATLAB: boundsParam.m and priorParam.m

This sets:
- parameter bounds
- proposal covariance scaling matrix (sigscale)
- prior families for each parameter
- mask vectors to indicate which parameters are fixed

All of this gets fed into the Metropolis-Hastings loop in main_erf.py.
"""

import numpy as np


def build_priors_and_bounds(YY, sigma2_1_fix):
    """
    Construct priors, masks, bounds, and proposal covariance scaling.

    Parameters
    ----------
    YY : ndarray (T,1)
        Excess market returns (used to set prior mean for mu_l).
        MATLAB uses mean(YY(:,1)) here.
    sigma2_1_fix : float
        Fixed variance in the expansion regime.

    Returns
    -------
    pshape : (7,) ndarray
        Prior family codes:
        1 = Beta, 2 = Gamma, 3 = Normal, 4 = InvGamma-ish, 5 = Uniform, 0 = none
    pmean : (7,) ndarray
        "Means" for priors (or lower bounds for Uniform).
    pstdd : (7,) ndarray
        "Stdevs" for priors (or upper bounds for Uniform).
    pmask : (7,) ndarray
        1 if parameter is FIXED (do not move in MH), else 0.
    pmaskinv : (7,) ndarray
        = 1 - pmask. 1 means free parameter.
    pfix : (7,) ndarray
        Values to pin fixed parameters to.
    lubound : (7,2) ndarray
        Lower and upper admissible bounds for each parameter.
        Order matches parameter vector:
        [mu_l, rho_l, corr_s, phi_1, phi_2, h, sigma2_1]
    sigscale : (7,7) ndarray
        Proposal covariance scaling matrix for MH jumps.
        Comes from the MATLAB code's 'sigscale' matrix.
    """
    mu_hat = float(np.mean(YY[:, 0]))

    # ---- Bounds ----
    # From your MATLAB snippet:
    # bmu    = [-1 1];
    # brho   = [-0.9999 0.9999];
    # bsigma = [0.00003 1];
    # bcorr  = [-0.9999 0.9999];
    # rsquared = [0 0.16];
    # hparam = [0, 5];
    #
    # lubound = [bmu;
    #            brho;
    #            brho;
    #            rsquared;
    #            rsquared;
    #            hparam;
    #            bsigma];
    #
    # Match parameter order:
    # [mu_l, rho_l, corr_s, phi_1, phi_2, h, sigma2_1]
    lubound = np.vstack([
        np.array([-1.0,      1.0]),        # mu_l
        np.array([-0.9999,   0.9999]),     # rho_l
        np.array([-0.9999,   0.9999]),     # corr_s (same bounds as rho_l in MATLAB)
        np.array([0.005,     0.16]),       # phi_1  (R^2-ish bound)
        np.array([0.0,       0.16]),       # phi_2
        np.array([0.0,       5.0]),        # h
        np.array([0.00003,   1.0])         # sigma2_1
    ])

    # ---- Priors ----
    # MATLAB 'prior' matrix columns:
    #   [ pshape, pmean, pstdd, pmask, pfix ]
    #
    # prior = [
    #   3,   mean(YY(:,1)), 0.0001, 0, mean(YY(:,1));   % mu_l  ~ Normal
    #   3,   0.97,          0.001,  0, 0.97;            % rho_l ~ Normal
    #   3,  -0.985,         0.05,   0, 0.99;            % corr_s ~ Normal ?
    #   5,   0,             0.2,    0, 0.99;            % phi_1  ~ Uniform(0,0.2)
    #   5,   0,             0.2,    0, 0.99;            % phi_2  ~ Uniform(0,0.2)
    #   5,   0,             4,      0, 0.99;            % h      ~ Uniform(0,4)
    #   5,   0,             4,      1, sigma2_1_fix];   % sigma2_1 fixed
    #
    # Note:
    # - The third row pfix=0.99 is odd but we'll keep it literally to
    #   fully replicate the MATLAB code.
    # - The uniform priors use [a,b] but MATLAB stored those as [pmean,pstdd].
    #   So for Uniform we interpret pmean = a, pstdd = b.

    pshape = np.array([3, 3, 3, 5, 5, 5, 5], dtype=float)

    pmean = np.array([
        mu_hat,   # mu_l
        0.97,     # rho_l
        -0.985,   # corr_s
        0.0,      # phi_1 lower bound
        0.0,      # phi_2 lower bound
        0.0,      # h lower bound
        0.0       # sigma2_1 lower bound (unused if fixed)
    ], dtype=float)

    pstdd = np.array([
        0.0001,   # mu_l prior std
        0.001,    # rho_l prior std
        0.05,     # corr_s prior std
        0.2,      # phi_1 upper bound (uniform)
        0.2,      # phi_2 upper bound (uniform)
        4.0,      # h upper bound (uniform)
        4.0       # sigma2_1 upper bound (uniform, but fixed anyway)
    ], dtype=float)

    pmask = np.array([
        0,    # mu_l free
        0,    # rho_l free
        0,    # corr_s free
        0,    # phi_1 free
        0,    # phi_2 free
        0,    # h free
        1     # sigma2_1 FIXED
    ], dtype=float)

    pfix = np.array([
        mu_hat,       # mu_l fixed value if pmask==1 (here it's not fixed, but we keep same structure)
        0.97,         # rho_l fixed value if needed
        0.99,         # corr_s "fixed" value used in MATLAB code
        0.99,         # phi_1 fallback
        0.99,         # phi_2 fallback
        0.99,         # h fallback
        sigma2_1_fix  # sigma2_1 is FIXED to expansion variance
    ], dtype=float)

    pmaskinv = 1.0 - pmask

    # ---- Proposal scaling (sigscale) ----
    # Directly from MATLAB:
    sigscale = np.array([
        [0.0006, 0,       0,       0,       0,       0,       0],
        [0,      0.0010,  0,       0,       0,       0,       0],
        [0,      0,       0.0010,  0,       0,       0,       0],
        [0,      0,       0,       0.0001,  0,       0,       0],
        [0,      0,       0,       0,       0.0001,  0,       0],
        [0,      0,       0,       0,       0,       0.1384,  0],
        [0,      0,       0,       0,       0,       0,       0.0002]
    ], dtype=float)

    return (
        pshape,
        pmean,
        pstdd,
        pmask,
        pmaskinv,
        pfix,
        lubound,
        sigscale,
    )


