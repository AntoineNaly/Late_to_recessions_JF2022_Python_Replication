# -*- coding: utf-8 -*-

"""
hamiltonfilter_xt_sv.py
------------------------
Hamilton filter and backward smoother for the two-state Markov chain S_t,
implementing the regime inference described in Section C.1 of the paper.

Forward pass — filtered probabilities π_{t|t} (eq. 16):
    For each t, compute P(S_t | F_t) by iterating Bayes' rule over the
    four (S_{t-1}, S_t) regime pairs.  At each step:
      1. Propagate: multiply the transition probability P(S_t | S_{t-1}) by
         the filtered marginal from the previous step.
      2. Weight by the Gaussian likelihood of the AR(1) innovation
         Ystar_t = z_t - φ_z*z_{t-1}, where the mean and variance depend on
         the regime pair through μ_z(S_t) and σ²_z(S_t) from eq. (14).
      3. Normalise to obtain the filtered probabilities P(S_t = s | F_t).

Backward pass — full-path draw p(S_{1:T} | F_T):
    Samples the complete regime path S_{1:T} using the Carter-Kohn recursion:
    starting from the filtered distribution at T, work backward and at each
    step draw S_t | S_{t+1}, F_t using Bayes' rule applied to the filtered
    probabilities and the Markov transition matrix.

The drawn path S_T is used in the Gibbs sampler to condition all other
parameter blocks on a single imputed regime sequence.

State encoding:  0 = recession,  1 = expansion

Performance:
    The forward and backward loops are JIT-compiled with Numba when available,
    using pre-drawn uniform samples to keep the RNG stream equivalent to the
    pure-NumPy fallback.  Falls back to vectorised NumPy when Numba is absent.
"""

import numpy as np

try:
    from .bingen import bingen as _bingen_fallback
except ImportError:
    try:
        from bingen import bingen as _bingen_fallback
    except ImportError:
        _bingen_fallback = None

# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
_NUMBA_AVAILABLE = False
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _hamilton_forward_nb(
        Ystar,          # (Tstar,)  AR(1) innovations z_t - φ_z*z_{t-1}
        mu_mat,         # (4, 2)    regime-pair means μ_z(S_{t-1}), μ_z(S_t)
        Sigma2_mat,     # (4,)      regime-pair variances σ²_z(S_t)
        pr_tr_vec,      # (4,)      column-major transition probs
        auxPhi,         # (2,)      [-phi_cc, 1]
        prob_1_vec,     # (4,)      initial joint probability
    ):
        """
        Forward Hamilton filter producing filtered marginal probabilities π_{t|t}.

        Returns
        -------
        fprob_raw : (2, Tstar)  filtered marginal probabilities (transposed
                                relative to the final output for cache efficiency)
        prob_1_vec_out : (4,)   final propagated joint probability (unused
                                externally but needed for numba return)
        """
        Tstar     = Ystar.shape[0]
        fprob_raw = np.zeros((2, Tstar))
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

        for t in range(Tstar):
            y_t = Ystar[t]

            # Innovation mean for each regime pair: Ystar_t - (μ_z(S_t) - φ_z*μ_z(S_{t-1}))
            # auxPhi = [-phi_cc, 1]  so  mu_mat @ auxPhi = μ_z(S_t) - φ_z*μ_z(S_{t-1})
            y0 = y_t - (mu_mat[0, 0] * auxPhi[0] + mu_mat[0, 1] * auxPhi[1])
            y1 = y_t - (mu_mat[1, 0] * auxPhi[0] + mu_mat[1, 1] * auxPhi[1])
            y2 = y_t - (mu_mat[2, 0] * auxPhi[0] + mu_mat[2, 1] * auxPhi[1])
            y3 = y_t - (mu_mat[3, 0] * auxPhi[0] + mu_mat[3, 1] * auxPhi[1])

            # Propagated joint prior: P(S_{t-1}, S_t) = P(S_t|S_{t-1}) * P(S_{t-1}|F_{t-1})
            d0 = pr_tr_vec[0] * prob_1_vec[0]
            d1 = pr_tr_vec[1] * prob_1_vec[1]
            d2 = pr_tr_vec[2] * prob_1_vec[2]
            d3 = pr_tr_vec[3] * prob_1_vec[3]

            # Likelihood × joint prior: N(Ystar_t; μ_pair, σ²_pair) * P(S_{t-1}, S_t)
            s0 = Sigma2_mat[0]; s1 = Sigma2_mat[1]
            s2 = Sigma2_mat[2]; s3 = Sigma2_mat[3]
            l0 = inv_sqrt_2pi / np.sqrt(s0) * np.exp(-0.5 * y0 * y0 / s0) * d0
            l1 = inv_sqrt_2pi / np.sqrt(s1) * np.exp(-0.5 * y1 * y1 / s1) * d1
            l2 = inv_sqrt_2pi / np.sqrt(s2) * np.exp(-0.5 * y2 * y2 / s2) * d2
            l3 = inv_sqrt_2pi / np.sqrt(s3) * np.exp(-0.5 * y3 * y3 / s3) * d3

            # Normalise to obtain filtered joint probabilities P(S_{t-1}, S_t | F_t)
            total = l0 + l1 + l2 + l3
            if total <= 0.0:
                l0 = 0.25; l1 = 0.25; l2 = 0.25; l3 = 0.25
            else:
                inv_t = 1.0 / total
                l0 *= inv_t; l1 *= inv_t; l2 *= inv_t; l3 *= inv_t

            # Marginalise over S_{t-1}: π_{t|t} = P(S_t | F_t)
            p_rec = l0 + l2    # P(S_t=0 | F_t): sum over pairs with S_t=recession
            p_exp = l1 + l3    # P(S_t=1 | F_t): sum over pairs with S_t=expansion

            fprob_raw[0, t] = p_rec
            fprob_raw[1, t] = p_exp

            # Propagate marginal for next step: P(S_t | F_t) enters next period's prior
            prob_1_vec[0] = p_rec
            prob_1_vec[1] = p_rec
            prob_1_vec[2] = p_exp
            prob_1_vec[3] = p_exp

        return fprob_raw

    @_njit(cache=True)
    def _hamilton_backward_nb(
        fprob,      # (Tstar, 2)  filtered probs π_{t|t}, row-major
        p,          # float  P(stay in recession | S_{t-1}=recession)
        q,          # float  P(stay in expansion | S_{t-1}=expansion)
        uniforms,   # (Tstar,) pre-drawn U(0,1) samples
    ):
        """
        Backward simulation: draws the full path S_{1:T} | F_T via Carter-Kohn.
        At each step: P(S_t | S_{t+1}, F_t) ∝ P(S_{t+1} | S_t) * π_{t|t}.
        """
        Tstar = fprob.shape[0]
        S_T   = np.zeros(Tstar, dtype=np.int64)

        # Initialise last state: at T, filtered = smoothed (no future data)
        p0 = fprob[Tstar - 1, 0]
        p1 = fprob[Tstar - 1, 1]
        S_T[Tstar - 1] = 0 if uniforms[Tstar - 1] * (p0 + p1) < p0 else 1

        # Backward recursion: P(S_t | S_{t+1}, F_t) ∝ P(S_{t+1}|S_t) * π_{t|t}
        for it in range(Tstar - 2, -1, -1):
            if S_T[it + 1] == 0:
                p0 = p         * fprob[it, 0]
                p1 = (1.0 - q) * fprob[it, 1]
            else:
                p0 = (1.0 - p) * fprob[it, 0]
                p1 = q         * fprob[it, 1]
            S_T[it] = 0 if uniforms[it] * (p0 + p1) < p0 else 1

        return S_T

    # Trigger JIT compilation on a tiny problem so first real call is instant
    def _warmup_numba_hamilton():
        _T  = 6
        _Ys = np.random.randn(_T)
        _mu = np.array([[0.0, 0.0], [0.0, 1.0],
                        [1.0, 0.0], [1.0, 1.0]])
        _s2 = np.ones(4)
        _pt = np.array([0.9, 0.1, 0.2, 0.8])
        _ap = np.array([-0.3, 1.0])
        _pv = np.array([0.25, 0.25, 0.25, 0.25])
        _fp = _hamilton_forward_nb(_Ys, _mu, _s2, _pt, _ap, _pv.copy())
        _fp_T = _fp.T.copy()   # (Tstar, 2)
        _u  = np.random.rand(_T)
        _hamilton_backward_nb(_fp_T, 0.9, 0.8, _u)

    _warmup_numba_hamilton()
    _NUMBA_AVAILABLE = True

except Exception:
    pass


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def hamiltonfilter_xt_sv(x_t, param_macro_MH, rng):
    """
    Parameters
    ----------
    x_t : (T,) ndarray
    param_macro_MH : dict
    rng : np.random.Generator

    Returns
    -------
    S_T   : (Tstar,) int ndarray  — backward-sampled regime path {0, 1}
    fprob : (Tstar, 2) float ndarray
    """

    x_t = np.asarray(x_t, dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Unpack parameters from eq. (14):
    #   z_{t+1} = μ_z(S_{t+1}) + φ_z(S_{t+1})(z_t - μ_z(S_t)) + σ_z(S_{t+1})ε_{z,t+1}
    #   μ_z(S_t) = μ_0 + μ_1 * S_t,   σ²_z(S_t) = σ²_{z,0}*(1 + h_z*S_t)
    # ------------------------------------------------------------------
    paramMU     = np.asarray(param_macro_MH["paramMU"], dtype=float)
    mu_0        = float(paramMU[0])                     # μ_0: mean of z_t in expansion (S_t=0)
    mu_1        = float(paramMU[1])                     # μ_1: increment so μ_z(1) = μ_0 + μ_1
    Sigma2_0_cc = float(param_macro_MH["Sigma2_0_cc"])  # σ²_{z,0}: baseline variance
    h_cc        = float(param_macro_MH["h_cc"])         # h_z: regime volatility shifter (=0 in practice)
    phi_cc      = float(param_macro_MH["phi_cc"])       # φ_z: AR(1) persistence from eq. (14)
    paramProb   = np.asarray(param_macro_MH["paramProb"], dtype=float)

    p = 1.0 - float(paramProb[0])     # P(stay in recession | currently recession)
    q = 1.0 - float(paramProb[1])     # P(stay in expansion | currently expansion)

    # ------------------------------------------------------------------
    # Enumerate the four (S_{t-1}, S_t) regime pairs
    # ------------------------------------------------------------------
    st_mat = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)

    # ------------------------------------------------------------------
    # Form the AR(1) innovation Ystar_t = z_t - φ_z*z_{t-1}
    # Under eq. (14): Ystar_t | (S_{t-1},S_t) ~ N(μ_z(S_t) - φ_z*μ_z(S_{t-1}), σ²_z(S_t))
    # ------------------------------------------------------------------
    Ystar = x_t[1:] - phi_cc * x_t[:-1]   # (Tstar,)
    Tstar = len(Ystar)

    # ------------------------------------------------------------------
    # Pre-compute regime-pair matrices
    # ------------------------------------------------------------------
    mu_mat     = mu_0 + st_mat * mu_1                          # (4, 2), μ_z(S) = μ_0 + μ_1*S for each pair
    Sigma2_mat = Sigma2_0_cc * (1.0 + h_cc * st_mat[:, 1])     # (4,), σ²_z(S_t) evaluated at S_t for each pair
    pr_tr      = np.array([[p, 1.0 - q], [1.0 - p, q]])        # 2x2 Markov transition matrix
    pr_tr_vec  = pr_tr.flatten(order='F')                      # [p,1-p,1-q,q] column-major
    auxPhi     = np.array([-phi_cc, 1.0])                      # mu_mat @ auxPhi = μ_z(S_t) - φ_z*μ_z(S_{t-1})

    # Steady-state distribution of the Markov chain as initialisation for π_{0|0}:
    # Stationary π satisfies π = P'π and π_0 + π_1 = 1.
    # Rearranged as (I - P')π = 0 with the normalisation constraint, solved via least-squares.
    A    = np.vstack([np.eye(2) - pr_tr, np.ones((1, 2))])
    EN   = np.array([0.0, 0.0, 1.0])
    pr_ss, *_ = np.linalg.lstsq(A.T @ A, A.T @ EN, rcond=None)
    if np.any(np.isnan(pr_ss)):
        pr_ss = np.array([0.5, 0.5])

    prob_0     = np.repeat(pr_ss, 2)
    prob_1_vec = pr_tr_vec * prob_0    # (4,) initial joint prior over regime pairs

    # ------------------------------------------------------------------
    # Forward Hamilton filter: compute π_{t|t} = P(S_t | F_t) for all t
    # ------------------------------------------------------------------
    if _NUMBA_AVAILABLE:
        # numba mutates prob_1_vec in-place — pass a copy
        fprob_raw = _hamilton_forward_nb(
            np.ascontiguousarray(Ystar),
            np.ascontiguousarray(mu_mat),
            np.ascontiguousarray(Sigma2_mat),
            np.ascontiguousarray(pr_tr_vec),
            np.ascontiguousarray(auxPhi),
            prob_1_vec.copy(),
        )
        fprob = np.ascontiguousarray(fprob_raw.T)   # (Tstar, 2)
    else:
        fprob = _hamilton_forward_numpy(
            Ystar, mu_mat, Sigma2_mat, pr_tr_vec, auxPhi, prob_1_vec
        )

    # ------------------------------------------------------------------
    # Backward simulation: draw S_{1:T} | F_T via Carter-Kohn recursion.
    # Pre-draw all Tstar uniforms at once, then pass into the JIT kernel.
    # ------------------------------------------------------------------
    uniforms = rng.uniform(size=Tstar)   # one vectorised RNG call

    if _NUMBA_AVAILABLE:
        S_T = _hamilton_backward_nb(
            np.ascontiguousarray(fprob),
            p, q,
            np.ascontiguousarray(uniforms),
        ).astype(int)
    else:
        S_T = _hamilton_backward_numpy(fprob, p, q, uniforms)

    return S_T, fprob


# ---------------------------------------------------------------------------
# NumPy fallbacks (used when numba is unavailable)
# ---------------------------------------------------------------------------

def _hamilton_forward_numpy(Ystar, mu_mat, Sigma2_mat, pr_tr_vec, auxPhi, prob_1_vec):
    Tstar = len(Ystar)
    fprob_raw = np.zeros((2, Tstar)) # π_{t|t}: filtered probability of each regime given all data up to time t
    for t in range(Tstar):
        y_t     = Ystar[t]
        y_error = y_t - (mu_mat @ auxPhi)   # Innovation mean per pair: Ystar_t - (μ_z(S_t) - φ_z*μ_z(S_{t-1}))
        prob_dd = pr_tr_vec * prob_1_vec    # Joint prior P(S_{t-1},S_t) = P(S_t|S_{t-1}) * P(S_{t-1}|F_{t-1})
        # Gaussian likelihood of Ystar_t under each regime pair (eq. 14):
        # f(Ystar_t | S_{t-1},S_t) = N(μ_z(S_t)-φ_z*μ_z(S_{t-1}), σ²_z(S_t))
        # Multiplying by prob_dd gives the joint: P(S_{t-1},S_t) * f(Ystar_t | S_{t-1},S_t)
        liki    = (
            (1.0 / np.sqrt(2.0 * np.pi * Sigma2_mat))
            * np.exp(-0.5 * y_error ** 2 / Sigma2_mat)
            * prob_dd
        )
        s = liki.sum()
        liki_adj = liki / s if s > 0.0 else np.ones(4) / 4.0
        # Marginalise over S_{t-1} to obtain π_{t|t} = P(S_t | F_t)
        prob_1  = liki_adj[:2] + liki_adj[2:]
        fprob_raw[:, t] = prob_1
        prob_1_vec = np.repeat(prob_1, 2)
    return fprob_raw.T   # (Tstar, 2)


def _hamilton_backward_numpy(fprob, p, q, uniforms):
    """Backward simulation of S_{1:T} | F_T using pre-drawn uniforms."""
    Tstar = fprob.shape[0]
    S_T   = np.zeros(Tstar, dtype=int)
    # At T, the filtered probability equals the smoothed probability (no future data)
    p0 = fprob[Tstar - 1, 0]    # P(S_T = recession | F_T)
    p1 = fprob[Tstar - 1, 1]    # P(S_T = expansion | F_T)
    S_T[Tstar - 1] = 0 if uniforms[Tstar - 1] * (p0 + p1) < p0 else 1
    for it in range(Tstar - 2, -1, -1):
        if S_T[it + 1] == 0:    # next period drawn as recession
            # P(S_t | S_{t+1}=rec, F_t) ∝ P(S_{t+1}=rec | S_t) * π_{t|t}
            p0 = p         * fprob[it, 0]   # P(rec→rec) * P(S_t=rec | F_t)
            p1 = (1.0 - q) * fprob[it, 1]  # P(exp→rec) * P(S_t=exp | F_t)
        else:                   # next period drawn as expansion
            # P(S_t | S_{t+1}=exp, F_t) ∝ P(S_{t+1}=exp | S_t) * π_{t|t}
            p0 = (1.0 - p) * fprob[it, 0]  # P(rec→exp) * P(S_t=rec | F_t)
            p1 = q         * fprob[it, 1]  # P(exp→exp) * P(S_t=exp | F_t)
        S_T[it] = 0 if uniforms[it] * (p0 + p1) < p0 else 1

    return S_T
