# -*- coding: utf-8 -*-

"""
utils.py
---------
Utility helpers for matrix algebra and numerical stability
used throughout the ExpectedReturnMeasure replication.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, svd

def solve_dlyap(F: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Wrapper for MATLAB's `dlyap` using SciPy's solver.

    Parameters
    ----------
    F : (n, n) array_like
        State transition matrix.
    Q : (n, n) array_like
        Covariance matrix of shocks.

    Returns
    -------
    P : (n, n) ndarray
        Solution to the discrete Lyapunov equation P = F P F' + Q.
    """
    return solve_discrete_lyapunov(F, Q)

def symmetrize(M: np.ndarray) -> np.ndarray:
    """Return (M + M.T)/2 for numerical symmetry."""
    return 0.5 * (M + M.T)

def chol_psd(M: np.ndarray) -> np.ndarray:
    """
    Compute a numerically safe Cholesky or SVD-based square root
    of a positive semidefinite matrix.
    """
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        U, s, _ = svd(M)
        return U @ np.diag(np.sqrt(np.clip(s, 0, None)))

def ensure_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator for reproducible draws."""
    return np.random.default_rng(seed)
