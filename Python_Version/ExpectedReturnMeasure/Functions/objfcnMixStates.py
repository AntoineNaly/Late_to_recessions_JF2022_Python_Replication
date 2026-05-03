# -*- coding: utf-8 -*-

"""
objfcnMixStates.py
------------------
Posterior objective function for the Metropolis-within-Gibbs sampler described
in Appendix B.4 of Gomez-Cram (2022, Journal of Finance).

Role in the estimation algorithm (Appendix B.4)
------------------------------------------------
Step 3 of the sampler draws the parameter vector Θ conditional on the
smoothed state sequence μ_{1:T} and observed excess returns y_{1:T}.
This function evaluates the log posterior,

    log p(Θ | μ_{1:T}, y_{1:T}) ∝ log p(y_{1:T} | Θ) + log p(Θ),   (19)

at a candidate draw from the Random-Walk Metropolis-Hastings proposal
N(Θ^{s−1}, c Ω).  The MH acceptance ratio is the difference of this
quantity evaluated at the candidate and the current draw.

Sign convention controlled by `indexMinimize`
---------------------------------------------
  indexMinimize == 0  (MH / Bayes mode):
      retp1 = log p(y|Θ) + log p(Θ)       [maximise]

  indexMinimize == 1  (optimiser mode, used by find_posterior_mode):
      retp1 = −log p(y|Θ) − log p(Θ)      [minimise to find mode Θ̃]
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm as norm_dist

try:
    from .evalmodMix import evalmod_mix
except ImportError:
    from evalmodMix import evalmod_mix


# ---------------------------------------------------------------------------
# Log-prior helper
# ---------------------------------------------------------------------------

def _log_prior_scalar(theta_i, shape_i, m_i, s_i):
    """
    Log prior density for a single free parameter, replicating MATLAB's
    objfcnMixStates.m prior block exactly.

    shape_i codes
    -------------
    1 = Beta(mean, std)      — bounded [0,1] parameters
    2 = Gamma(mean, std)     — positive parameters
    3 = Normal(mean, std)    — unconstrained parameters
    4 = Inverse-Gamma-like   — custom kernel from MATLAB code
    5 = Uniform(a, b)        — flat; NOTE: replicates MATLAB's log CDF form
    0 = flat (no contribution)
    """
    if shape_i == 0:
        return 0.0

    if shape_i == 1:
        a = (1.0 - m_i) * (m_i ** 2) / (s_i ** 2) - m_i
        b = a * (1.0 / m_i - 1.0)
        return np.log(beta_dist.pdf(theta_i, a, b))

    if shape_i == 2:
        scale = (s_i ** 2) / m_i
        shape = m_i / scale
        return np.log(gamma_dist.pdf(theta_i, shape, scale=scale))

    if shape_i == 3:
        # logpdf avoids log(0) warnings near the tails
        return norm_dist.logpdf(theta_i, loc=m_i, scale=s_i)

    if shape_i == 4:
        aux = theta_i ** 2
        a, b = m_i, s_i
        val = (aux ** (-b - 1.0)) * np.exp(-0.5 * b * (a ** 2) / (aux ** 2))
        return np.log(val) if val > 0 else -np.inf

    if shape_i == 5:
        # Replicates MATLAB exactly: log((para-a)/(b-a))
        # Note: this is the log CDF, not the log PDF — intentional to match MATLAB.
        a, b = m_i, s_i
        if a < theta_i < b:
            return np.log((theta_i - a) / (b - a))
        return -np.inf

    return 0.0


# ---------------------------------------------------------------------------
# Main objective
# ---------------------------------------------------------------------------

def objfcn_mix_states(
    para,
    YY,
    pi_t,
    indexMinimize,
    pshape,
    pmean,
    pstdd,
    pmask,
    pmaskinv,
    pfix,
    lubound,
    rng=None,
):
    """
    Evaluate log p(Θ | data) at the candidate parameter vector `para`.

    Fixed parameters (pmask[i]==1) are substituted from pfix before the
    likelihood is evaluated.  If any free parameter is outside lubound the
    function returns the sentinel value −10²⁰ so the MH step rejects.

    Parameters
    ----------
    para       : (n,) array   — candidate parameter vector
    YY         : (T, 1) array — observed excess returns r^e_{1:T}  (eq. 2)
    pi_t       : (T,) array   — filtered recessionary probabilities π̂_{t|t}
    indexMinimize : int       — 0 = MH mode, 1 = optimiser mode
    pshape, pmean, pstdd : (n,) arrays — prior specification
    pmask, pmaskinv, pfix    : (n,) arrays — fixed-parameter mask
    lubound    : (n, 2) array — lower/upper admissible bounds
    rng        : np.random.Generator or None

    Returns
    -------
    retp1 : float         — posterior objective (sign per indexMinimize)
    retp2 : float         — log-likelihood only
    At_draw_tot, At_mat_tot, At_pred_tot : (T, mdim) arrays — state sequences
    Kgain       : (2,) array
    loglh_tot   : (T,) array
    modelInfo_1, modelInfo_2 : (T, 4) arrays
    """
    if rng is None:
        rng = np.random.default_rng()

    para     = np.asarray(para,     dtype=float)
    pmask    = np.asarray(pmask,    dtype=float)
    pmaskinv = np.asarray(pmaskinv, dtype=float)
    pfix     = np.asarray(pfix,     dtype=float)
    pshape   = np.asarray(pshape,   dtype=float)
    pmean    = np.asarray(pmean,    dtype=float)
    pstdd    = np.asarray(pstdd,    dtype=float)
    low      = lubound[:, 0]
    high     = lubound[:, 1]

    # Bounds check — fixed parameters are exempt (pmask adds 1 to their check)
    parabd_ind1 = np.prod((para > low)  + pmask)
    parabd_ind2 = np.prod((para < high) + pmask)

    if (parabd_ind1 > 0) and (parabd_ind2 > 0):

        modelpara = para * pmaskinv + pfix * pmask

        (
            loglh_tot,
            At_draw_tot,
            At_mat_tot,
            At_pred_tot,
            Kgain,
            modelInfo_1,
            modelInfo_2,
        ) = evalmod_mix(modelpara, YY, pi_t, indexMinimize, rng=rng)

        lnpY = float(np.sum(loglh_tot))

        lnprio = 0.0
        for i in range(len(para)):
            if pmask[i] == 1:
                continue
            lnprio += _log_prior_scalar(
                para[i], int(pshape[i]), pmean[i], pstdd[i]
            )

        if indexMinimize == 1:
            retp1 = float(np.real(lnpY - lnprio))
        else:
            retp1 = float(np.real(lnpY + lnprio))

        retp2 = float(np.real(lnpY))

    else:
        sentinel     = -1e20
        retp1        = sentinel
        retp2        = sentinel
        At_draw_tot  = sentinel
        At_mat_tot   = sentinel
        At_pred_tot  = sentinel
        Kgain        = np.zeros(2)
        loglh_tot    = np.array([sentinel])
        modelInfo_1  = np.full((1, 4), sentinel)
        modelInfo_2  = np.full((1, 4), sentinel)

    return (
        retp1, retp2,
        At_draw_tot, At_mat_tot, At_pred_tot,
        Kgain, loglh_tot,
        modelInfo_1, modelInfo_2,
    )
