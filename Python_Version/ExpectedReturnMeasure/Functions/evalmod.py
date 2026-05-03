# -*- coding: utf-8 -*-

"""
evalmod.py
----------
Single-regime Kalman filter and simulation smoother for the state-space
model of excess returns described in Appendix B.1.2 (Gomez-Cram 2022).

State-space representation (eq. B.6 / Appendix B.1.2)
------------------------------------------------------
Measurement:    y_{t+1} = H0 + H1 · x_{t+1}
Transition:     x_{t+1} = F0 + F1 · x_t + η_{t+1},   η_{t+1} ~ N(0, Q(S_t))

where the latent state is x_t = [μ_{t+1}, μ_t, σ_r(S_t) ε^r_{t+1}]'
(three-dimensional, mdim=3) and y_t = r^e_t is the scalar excess return.

Speed design
------------
The two inner loops (forward filter over t=1…T, backward smoother over
t=T…1) are compiled to native code with Numba @njit when Numba is available.
Random numbers for the smoother are generated outside the JIT kernel so the
existing numpy.random.Generator RNG is used for reproducibility.

If Numba is not installed the same functions run as plain Python — correct
but slower.  Install with:  conda install numba   or   pip install numba
The first Numba-enabled call triggers JIT compilation (~15 s); subsequent
calls load from cache and are fast.
"""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

try:
    from .coefficients import coefficients
except ImportError:
    from coefficients import coefficients

# ---------------------------------------------------------------------------
# Numba import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import numba

    def _njit(**kwargs):
        return numba.njit(**kwargs)

    NUMBA_AVAILABLE = True

except ImportError:

    def _njit(**kwargs):
        """Identity decorator — returns the function unchanged."""
        def decorator(fn):
            return fn
        return decorator

    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# JIT kernel 1 — Forward Kalman filter
# ---------------------------------------------------------------------------

@_njit(cache=True, nogil=True)
def _forward_filter(YY_1d, H0_s, H1_v, RR_s, F0_v, F1_m, Q_m, At0, Pt0):
    """
    Forward Kalman filter for a scalar measurement (nv = 1).

    Implements the standard predict-update recursion (Appendix B.1.2):
        Predict:  α̂_{t|t-1} = F0 + F1 · α_{t-1|t-1}
                  P_{t|t-1}  = F1 · P_{t-1|t-1} · F1' + Q
        Update:   ν_t = y_t − H0 − H1 · α̂_{t|t-1}
                  F_t = H1 · P_{t|t-1} · H1' + R
                  K_t = P_{t|t-1} · H1' / F_t
                  α_{t|t} = α̂_{t|t-1} + K_t · ν_t
                  P_{t|t} = P_{t|t-1} − K_t · F_t · K_t'

    Log-likelihood contribution (standard Gaussian):
        log p(y_t | y_{1:t-1}) = −½ log(2π) − ½ log F_t − ½ ν_t² / F_t

    Parameters (all pre-flattened for Numba compatibility)
    ----------
    YY_1d  : (T,)           — observed excess returns r^e_{1:T}
    H0_s   : scalar         — measurement intercept
    H1_v   : (mdim,)        — measurement row vector (H1 in eq. B.6)
    RR_s   : scalar         — measurement noise variance R
    F0_v   : (mdim,)        — state transition intercept F0
    F1_m   : (mdim, mdim)   — state transition matrix F1
    Q_m    : (mdim, mdim)   — state noise covariance Q(S_t)
    At0    : (mdim,)        — initial state mean α_{1|0}
    Pt0    : (mdim, mdim)   — initial state covariance P_{1|0}

    Returns
    -------
    loglh       : (T,)             per-period log p(y_t | y_{1:t-1})
    At_out      : (T, mdim)        filtered means α_{t|t}
    Pt_out      : (T, mdim, mdim)  filtered covariances P_{t|t}
    Kg_out      : (T, mdim)        Kalman gains K_t
    At_pred_out : (T, mdim)        predicted means α_{t|t-1}
    """
    T    = len(YY_1d)
    mdim = len(F0_v)

    At_pred_out = np.zeros((T, mdim))
    Pt_out      = np.zeros((T, mdim, mdim))
    Kg_out      = np.zeros((T, mdim))
    At_out      = np.zeros((T, mdim))
    loglh       = np.zeros(T)

    HALF_LOG2PI = 0.9189385332046728   # 0.5 * log(2π)

    At = At0.copy()
    Pt = Pt0.copy()

    for t in range(T):

        # Prediction step
        alphahat = F0_v + F1_m @ At
        Phat     = F1_m @ Pt @ F1_m.T + Q_m
        Phat     = 0.5 * (Phat + Phat.T)

        # Scalar innovation (nv = 1 → Ft and nut are scalars)
        yhat   = H0_s + np.dot(H1_v, alphahat)
        nut    = YY_1d[t] - yhat
        PhatH1 = Phat @ H1_v                       # (mdim,)
        Ft     = np.dot(H1_v, PhatH1) + RR_s       # scalar innovation variance
        invFt  = 1.0 / Ft

        # Log-likelihood contribution
        loglh[t] = -HALF_LOG2PI - 0.5 * np.log(Ft) - 0.5 * nut * nut * invFt

        # Kalman gain and state update
        Kgain = PhatH1 * invFt
        At    = alphahat + Kgain * nut
        Pt    = Phat - np.outer(PhatH1, PhatH1) * invFt
        Pt    = 0.5 * (Pt + Pt.T)

        At_out[t]       = At
        Pt_out[t]       = Pt
        Kg_out[t]       = Kgain
        At_pred_out[t]  = alphahat

    return loglh, At_out, Pt_out, Kg_out, At_pred_out


# ---------------------------------------------------------------------------
# JIT kernel 2 — Backward simulation smoother
# ---------------------------------------------------------------------------

@_njit(cache=True, nogil=True)
def _backward_smoother(At_out, Pt_out, F0_v, F1_m, Q_m, z_T, z_back):
    """
    Rauch-Tung-Striebel simulation smoother (Appendix B.4, step 2).

    Draws a sample path x_{1:T} | y_{1:T} by working backwards from the
    terminal filtered distribution:

        x_T | y_{1:T}           ~ N(α_{T|T}, P_{T|T})
        x_t | x_{t+1}, y_{1:T} ~ N(α̃_t, P̃_t)   for t = T-1, …, 1

    where the backward mean and variance are:
        J_t   = P_{t|t} · F1' · P_{t+1|t}^{-1}
        α̃_t  = α_{t|t} + J_t · (x_{t+1} − α̂_{t+1|t})
        P̃_t  = P_{t|t} − J_t · P_{t+1|t} · J_t'

    The matrix square root uses SVD to handle near-singular covariances:
        P = U diag(s) V'  →  L = U diag(√s),   draw = mean + L @ z

    Pre-generated random arrays (z_T, z_back) are passed in from the Python
    wrapper so the numpy.random.Generator state is advanced outside Numba.

    Parameters
    ----------
    At_out : (T, mdim)          filtered means from _forward_filter
    Pt_out : (T, mdim, mdim)    filtered covariances from _forward_filter
    F0_v   : (mdim,)            state transition intercept
    F1_m   : (mdim, mdim)       state transition matrix
    Q_m    : (mdim, mdim)       state noise covariance
    z_T    : (mdim,)            N(0,1) draw for terminal period
    z_back : (T-1, mdim)        N(0,1) draws for backward pass

    Returns
    -------
    At_draw : (T, mdim)  — sampled state path x_{1:T} | y_{1:T}
    """
    T, mdim = At_out.shape
    At_draw = np.zeros((T, mdim))

    # --- Terminal draw ~ N(α_{T|T}, P_{T|T}) ---
    Pt_T    = Pt_out[T - 1].copy()
    U, s, _ = np.linalg.svd(Pt_T)
    sqrt_s  = np.sqrt(np.maximum(s, 0.0))
    L = np.zeros((mdim, mdim))
    for j in range(mdim):
        for i in range(mdim):
            L[i, j] = U[i, j] * sqrt_s[j]
    At_draw[T - 1] = At_out[T - 1] + L @ z_T

    # --- Backward pass ---
    for back in range(1, T):
        tb = T - 1 - back

        Att = At_out[tb]
        Ptt = Pt_out[tb].copy()

        # One-step-ahead predictive covariance
        Phat    = F1_m @ Ptt @ F1_m.T + Q_m
        Phat    = 0.5 * (Phat + Phat.T)
        invPhat = np.linalg.inv(Phat)

        # Backward innovation: deviation of smoothed draw from prediction
        nut_b  = At_draw[tb + 1] - (F0_v + F1_m @ Att)
        PttF1T = Ptt @ F1_m.T                    # P_{t|t} · F1'

        # Backward mean and covariance
        Amean = Att + PttF1T @ invPhat @ nut_b
        Pmean = Ptt - PttF1T @ invPhat @ PttF1T.T
        Pmean = 0.5 * (Pmean + Pmean.T)

        # Draw ~ N(Amean, Pmean)
        Um, sm, _ = np.linalg.svd(Pmean)
        sqrt_sm   = np.sqrt(np.maximum(sm, 0.0))
        Lm = np.zeros((mdim, mdim))
        for j in range(mdim):
            for i in range(mdim):
                Lm[i, j] = Um[i, j] * sqrt_sm[j]

        At_draw[tb] = Amean + Lm @ z_back[back - 1]

    return At_draw


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def evalmod(para, YY, indexMinimize, rng=None):
    """
    State-space filter and smoother for a single economic regime.

    Sets up the system matrices from structural parameters via coefficients(),
    solves for the unconditional initial covariance (discrete Lyapunov
    equation), then dispatches to the JIT-compiled kernels (or plain Python
    equivalents if Numba is not installed).

    Parameters
    ----------
    para : (5,) array-like
        [μ_0, ρ, ρ_{μ,r}, φ, σ²]
        Structural parameters for this regime (expansion or recession).
        φ and σ² differ between regimes; μ_0, ρ, ρ_{μ,r} are shared
        (Appendix B.1.2).
    YY : (T, 1) ndarray
        Observed excess market returns r^e_{1:T}  (eq. 2).
    indexMinimize : int
        0 → return log-likelihood and run simulation smoother (MH mode).
        1 → return negative log-likelihood, skip smoother (optimiser mode).
    rng : np.random.Generator or None

    Returns
    -------
    loglh   : (T,)       per-period ±log p(y_t | y_{1:t-1})
    At_draw : (T, mdim)  smoothed state draw (zeros if indexMinimize==1)
    At_mat  : (T, mdim)  filtered means α_{t|t}
    Kg_mat  : (T, mdim)  Kalman gains K_t
    At_pred : (T, mdim)  predicted means α_{t|t-1}
    """
    if rng is None:
        rng = np.random.default_rng()

    YY   = np.asarray(YY, dtype=float)
    T, _ = YY.shape
    para = np.asarray(para, dtype=float)

    # Build system matrices from structural parameters (Appendix B.1.2)
    H0, H1, RR, F0, F1, Q = coefficients(para)
    mdim = F1.shape[0]

    # Initial conditions: prior mean α_{1|0} and unconditional covariance P_{1|0}.
    # P_{1|0} solves the discrete Lyapunov equation: P = F1·P·F1' + Q.
    At0 = np.array([para[0], 0.0, 0.0], dtype=float)
    Pt0 = solve_discrete_lyapunov(F1, Q)

    # Flatten to 1D / scalar for Numba (avoids ambiguous broadcasting)
    H0_s  = float(np.atleast_1d(H0).flat[0])
    H1_v  = np.ascontiguousarray(np.atleast_1d(H1).reshape(-1), dtype=float)
    RR_s  = float(np.atleast_1d(RR).flat[0])
    F0_v  = np.ascontiguousarray(np.atleast_1d(F0).reshape(-1), dtype=float)
    F1_m  = np.ascontiguousarray(F1,             dtype=float)
    Q_m   = np.ascontiguousarray(Q,              dtype=float)
    YY_1d = np.ascontiguousarray(YY[:, 0],       dtype=float)

    # Forward filter (JIT-compiled or plain Python)
    loglh, At_mat, Pt_mat, Kg_mat, At_pred = _forward_filter(
        YY_1d, H0_s, H1_v, RR_s, F0_v, F1_m, Q_m, At0, Pt0
    )

    if indexMinimize == 1:
        # Optimiser mode: return negative log-likelihood, skip smoother
        return -loglh, np.zeros((T, mdim)), At_mat, Kg_mat, At_pred

    # Simulation smoother — draw random numbers outside Numba so the
    # numpy.random.Generator state is consumed in the Python layer
    z_T    = rng.standard_normal(mdim)
    z_back = rng.standard_normal((T - 1, mdim))

    At_draw = _backward_smoother(At_mat, Pt_mat, F0_v, F1_m, Q_m, z_T, z_back)

    return loglh, At_draw, At_mat, Kg_mat, At_pred


def warmup_jit():
    """
    Pre-compile and cache both JIT kernels.

    Call once at the start of a session (takes ~15 s on the very first ever
    run; subsequent calls load from cache instantly).  After this, all
    evalmod calls run at native speed.

    Has no effect if Numba is not installed.
    """
    if not NUMBA_AVAILABLE:
        print("Numba not available — running in pure Python mode (slower).\n"
              "Install with:  conda install numba   or   pip install numba")
        return

    mdim, T_dummy = 3, 10
    rng = np.random.default_rng(0)

    dummy_Y  = np.zeros(T_dummy)
    dummy_H1 = np.zeros(mdim)
    dummy_F1 = np.eye(mdim)
    dummy_Q  = np.eye(mdim) * 0.01
    dummy_F0 = np.zeros(mdim)
    dummy_Pt = np.eye(mdim) * 0.01
    dummy_At = np.zeros(mdim)

    loglh, At_out, Pt_out, Kg, At_pred = _forward_filter(
        dummy_Y, 0.0, dummy_H1, 0.01, dummy_F0, dummy_F1, dummy_Q,
        dummy_At, dummy_Pt,
    )
    _backward_smoother(
        At_out, Pt_out, dummy_F0, dummy_F1, dummy_Q,
        rng.standard_normal(mdim),
        rng.standard_normal((T_dummy - 1, mdim)),
    )
    print("Numba JIT kernels compiled and cached.")
