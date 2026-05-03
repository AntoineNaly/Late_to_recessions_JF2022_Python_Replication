# -*- coding: utf-8 -*-

"""
bingen.py
---------
Draws binary regime states {0, 1} from unnormalised weights (p0, p1).

Used in the Hamilton filter backward smoother to draw S_t | S_{t+1}, F_t
from the unnormalised pair (p0, p1) where:
    p0 = P(S_{t+1} | S_t=0) * π_{t|t}(0)
    p1 = P(S_{t+1} | S_t=1) * π_{t|t}(1)

The draw is equivalent to:  S_t = 0  if  U < p0/(p0+p1),  else  S_t = 1
where U ~ Uniform(0,1).  This implements the Bernoulli step in the Carter-Kohn
backward simulation of the Markov chain for eq. (14).
"""

import numpy as np


def bingen(p0: float, p1: float, m: int, rng: np.random.Generator | None = None):
    """
    Draw binary states given unnormalized weights p0, p1.

    Parameters
    ----------
    p0 : float
        Weight associated with state 0.
    p1 : float
        Weight associated with state 1.
    m : int
        Number of draws to generate.
    rng : np.random.Generator or None
        RNG to use. If None, a default generator will be created.

    Returns
    -------
    s : ndarray of shape (m,)
        Draws in {0,1}, where P(s==0) = p0 / (p0+p1).
    """
    if rng is None:
        rng = np.random.default_rng()

    denom = p0 + p1
    if denom <= 0:
        # Degenerate but guard anyway
        pr0 = 0.5
    else:
        pr0 = p0 / denom

    u = rng.random(m)
    # P(s=0) = pr0: draw 0 if u < pr0, else draw 1
    s = 1 - (u < pr0).astype(int)
    return s
