# -*- coding: utf-8 -*-

"""
evalmod_mix.py
--------------
Regime-switching mixture likelihood for the excess return model (eq. 5,
Gomez-Cram 2022).

Mixture predictive density  (eq. 5)
------------------------------------
    p(r^e_{t+1} | Θ, r^e_{1:t}) =
        (1 − π̂_{t|t}) · p(r^e_{t+1} | Θ_Exp, r^e_{1:t})
      +     π̂_{t|t}  · p(r^e_{t+1} | Θ_Rec, r^e_{1:t})

where π̂_{t|t} is the filtered recessionary probability from the first-stage
business-cycle model (Appendix B.3) and the two regime densities come from
the single-regime Kalman filter in evalmod.py.

Regime parameters (Appendix B.1.2)
------------------------------------
Expansion:  para_1 = [μ_0, ρ, ρ_{μ,r}, φ_1,        σ²_1]
Recession:  para_2 = [μ_0, ρ, ρ_{μ,r}, φ_2,  σ²_1·(1+h)]

μ_0, ρ, ρ_{μ,r} are shared; φ and σ² scale the noise covariance Q(S_t)
and differ across regimes (eq. 4, Σ(S_t) block).

Parallelism
-----------
When Numba is available, its @njit kernels release the GIL during execution.
A module-level ThreadPoolExecutor with 2 workers therefore runs both regime
filters simultaneously, giving close to 2× speedup on any multi-core machine.
Two independent child RNGs are derived from the parent so thread-safety is
guaranteed while the chain remains reproducible given a fixed seed.

When Numba is not available, threading is skipped — Python's GIL means
threads would add overhead with no gain, so the two regimes run sequentially.
"""

import numpy as np

try:
    from .evalmod import evalmod, NUMBA_AVAILABLE
except ImportError:
    from evalmod import evalmod, NUMBA_AVAILABLE

# Thread pool is only beneficial when Numba releases the GIL.
# In pure Python mode, the GIL serialises threads anyway, so we skip it.
if NUMBA_AVAILABLE:
    from concurrent.futures import ThreadPoolExecutor
    _POOL = ThreadPoolExecutor(max_workers=2)


def evalmod_mix(para, YY, pi_t, indexMinimize, rng=None):
    """
    Mixture Kalman filter across expansion and recession regimes.

    Parameters
    ----------
    para : (7,) array-like
        [μ_0, ρ, ρ_{μ,r}, φ_1, φ_2, h, σ²_1]
    YY : (T, 1) ndarray
        Observed excess returns r^e_{1:T}  (eq. 2).
    pi_t : (T,) ndarray
        Filtered recessionary probabilities π̂_{t|t}  (eq. 5).
    indexMinimize : int
        Forwarded to evalmod (0 = MH mode, 1 = optimiser mode).
    rng : np.random.Generator or None

    Returns
    -------
    loglh_tot   : (T,)      mixture log-likelihood per period
    At_draw_tot : (T, mdim) mixture smoothed state draw
    At_mat_tot  : (T, mdim) mixture filtered mean
    At_pred_tot : (T, mdim) mixture predicted mean
    Kgain       : (2,)      first-period Kalman gain per regime (diagnostics)
    modelInfo_1 : (T, 4)    [loglh_1, At_draw_1[:,1], At_mat_1[:,1], At_pred_1[:,1]]
    modelInfo_2 : (T, 4)    same for recession regime
    """
    if rng is None:
        rng = np.random.default_rng()

    para = np.asarray(para, dtype=float)
    mu_l, rho_l, corr_s, phi_1, phi_2, h, sigma2_1 = para

    # Regime-specific parameter vectors (Appendix B.1.2)
    para_1 = np.array([mu_l, rho_l, corr_s, phi_1, sigma2_1],              dtype=float)
    para_2 = np.array([mu_l, rho_l, corr_s, phi_2, sigma2_1 * (1.0 + h)], dtype=float)

    # Two independent child RNGs — one per regime.
    # Derived from the parent RNG so the overall chain is reproducible.
    child_seeds = rng.integers(0, 2**32, size=2)
    rng1 = np.random.default_rng(int(child_seeds[0]))
    rng2 = np.random.default_rng(int(child_seeds[1]))

    if NUMBA_AVAILABLE:
        # Parallel path: submit both regime evaluations to the thread pool.
        # Numba's @njit kernels release the GIL, giving true parallelism.
        f1 = _POOL.submit(evalmod, para_1, YY, indexMinimize, rng1)
        f2 = _POOL.submit(evalmod, para_2, YY, indexMinimize, rng2)
        loglh_1, At_draw_1, At_mat_1, Kg_mat_1, At_pred_1 = f1.result()
        loglh_2, At_draw_2, At_mat_2, Kg_mat_2, At_pred_2 = f2.result()
    else:
        # Sequential fallback: threading would only add overhead here
        # since the GIL prevents parallel Python execution.
        loglh_1, At_draw_1, At_mat_1, Kg_mat_1, At_pred_1 = evalmod(
            para_1, YY, indexMinimize, rng1
        )
        loglh_2, At_draw_2, At_mat_2, Kg_mat_2, At_pred_2 = evalmod(
            para_2, YY, indexMinimize, rng2
        )

    # Mixture weights: (1−π̂_{t|t}) for expansion, π̂_{t|t} for recession
    pi = np.asarray(pi_t, dtype=float).reshape(-1, 1)
    w1 = 1.0 - pi    # (T, 1) expansion weight
    w2 = pi          # (T, 1) recession weight

    # Mixture outputs (eq. 5 applied to states and likelihoods)
    loglh_tot   = (w1 * loglh_1.reshape(-1, 1) + w2 * loglh_2.reshape(-1, 1)).reshape(-1)
    At_draw_tot = w1 * At_draw_1 + w2 * At_draw_2
    At_mat_tot  = w1 * At_mat_1  + w2 * At_mat_2
    At_pred_tot = w1 * At_pred_1 + w2 * At_pred_2

    # First-period Kalman gains for each regime (diagnostics)
    Kgain = np.array([Kg_mat_1[0, 0], Kg_mat_2[0, 0]], dtype=float)

    # Auxiliary output: state index 1 (μ_t, the conditional expected return)
    # is stored in column 1 (0-indexed) of the state vector x_t.
    col = 1
    modelInfo_1 = np.column_stack([
        loglh_1, At_draw_1[:, col], At_mat_1[:, col], At_pred_1[:, col]
    ])
    modelInfo_2 = np.column_stack([
        loglh_2, At_draw_2[:, col], At_mat_2[:, col], At_pred_2[:, col]
    ])

    return loglh_tot, At_draw_tot, At_mat_tot, At_pred_tot, Kgain, modelInfo_1, modelInfo_2
