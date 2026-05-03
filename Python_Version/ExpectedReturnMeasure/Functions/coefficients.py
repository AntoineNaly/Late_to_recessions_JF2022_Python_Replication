# -*- coding: utf-8 -*-

"""
coefficients.py
----------------
Translated from MATLAB: coefficients.m

Computes the state-space matrices (H0, H1, RR, F0, F1, Q)
used in the dynamic factor model with correlated errors.
"""

import numpy as np

def coefficients(para: np.ndarray):
    """
    Compute the coefficient matrices of the state-space model.

    Parameters
    ----------
    para : array_like of shape (5,)
        Model parameters [mu_l, rho_l, corr_s, phi, sigma2].

    Returns
    -------
    H0 : float
        Intercept in the measurement equation.
    H1 : ndarray (1,3)
        Coefficient vector for latent states in measurement equation.
    RR : float
        Measurement error variance (set to zero in baseline model).
    F0 : ndarray (3,1)
        Constant term in the transition equation.
    F1 : ndarray (3,3)
        State-transition matrix.
    Q : ndarray (3,3)
        Covariance matrix of innovations.
    """
    mu_l, rho_l, corr_s, phi, sigma2 = para

    # Variances following the authors' structure
    sigma2_l  = (1 - rho_l**2) * phi * sigma2
    sigma2_r  = (1 - phi) * sigma2
    sigma2_lr = corr_s * sigma2 * np.sqrt((1 - rho_l**2) * phi * (1 - phi))

    # Measurement matrices
    H0 = 0.0
    H1 = np.array([[0.0, 1.0, 1.0]])
    RR = 0.0

    # State transition components
    F0 = np.array([[mu_l * (1 - rho_l)], [0.0], [0.0]])
    F1 = np.array([[rho_l, 0.0, 0.0],
                   [1.0,   0.0, 0.0],
                   [0.0,   0.0, 0.0]])

    # Innovation covariance
    Q = np.array([[sigma2_l, 0.0, sigma2_lr],
                  [0.0,      0.0, 0.0],
                  [sigma2_lr,0.0, sigma2_r]])

    return H0, H1, RR, F0, F1, Q


