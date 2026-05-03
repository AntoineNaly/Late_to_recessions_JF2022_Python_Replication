# -*- coding: utf-8 -*-

"""
utils.py
--------
Shared utilities for the StateBusinessCycle replication.

This includes:
- RNG helpers
- Safe Cholesky / SVD square-root for PSD matrices
- OLS helper for intercept-free regression (MATLAB fitlm(...,'intercept',false))
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import svd, lstsq, LinAlgError


def ensure_rng(seed: int | None = None) -> np.random.Generator:
    """
    Create a reproducible NumPy random Generator.

    Parameters
    ----------
    seed : int or None
        If provided, use this seed. If None, use default entropy.

    Returns
    -------
    rng : np.random.Generator
    """
    return np.random.default_rng(seed)


def chol_psd(M: np.ndarray) -> np.ndarray:
    """
    Return a "square root" factor C such that C @ C.T ≈ M for a PSD matrix.

    We first try Cholesky. If that fails (semi-definite), we fall back to
    SVD-based sqrt.

    Parameters
    ----------
    M : (n,n) array_like
        Symmetric positive semidefinite matrix.

    Returns
    -------
    C : (n,n) ndarray
        Lower-ish factor.
    """
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        U, s, _ = svd(M)
        return U @ np.diag(np.sqrt(np.clip(s, 0, None)))


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T)/2 to kill asymmetry from numerics."""
    return 0.5 * (A + A.T)


def ols_no_intercept(X: np.ndarray, y: np.ndarray):
    """
    Solve y ≈ X @ beta via OLS with *no intercept*, matching MATLAB's
    fitlm(y ~ X, 'intercept', false).

    Parameters
    ----------
    X : (T,k) array
        Regressor matrix.
    y : (T,) or (T,1) array
        Dependent variable.

    Returns
    -------
    beta_hat : (k,) ndarray
        Estimated slope coefficients.
    resid : (T,) ndarray
        Residuals y - X @ beta_hat.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Solve least squares: minimize ||X b - y||^2
    beta_hat, *_ = lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    return beta_hat, resid
