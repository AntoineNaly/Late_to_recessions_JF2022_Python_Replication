# -*- coding: utf-8 -*-

"""
generate_ChangeState.py
-----------------------
Counts observed transitions between discrete Markov states in the imputed
regime path S_{1:T}.

The Markov chain for S_t (eq. 14) has a 2×2 transition matrix:
    P = [[P(0→0), P(0→1)],    P(stay in recession), P(leave recession)
         [P(1→0), P(1→1)]]    P(leave expansion),   P(stay in expansion)

The Beta posteriors for the transition probabilities p and q are conjugate
to the Binomial likelihood, so the sufficient statistics are the transition
counts from the imputed path S_{1:T}.  This function computes those counts:

    change_state[i, j] = #{t : S_{t-1} = states[i],  S_t = states[j]}

These counts are added to the Beta prior pseudo-counts U1_00_, U1_01_, U1_10_
in initial_values_macro (and in the main Gibbs loop) to obtain the posterior
Beta parameters for p and q.
"""

import numpy as np


def generate_change_state(S_T: np.ndarray, states: np.ndarray | list[int]):
    """
    Count transitions between discrete states.

    Parameters
    ----------
    S_T : array-like of shape (T,)
        Sequence of states taking values in {1,2} (1=expansion, 2=recession
        per the calling convention).
    states : array-like
        Unique sorted state labels (e.g. [1,2]).

    Returns
    -------
    change_state : (m,m) ndarray
        Matrix where entry (i,j) is the count of transitions
        state_i -> state_j.
    """
    S_T    = np.asarray(S_T,    dtype=np.int64).reshape(-1)
    states = np.asarray(states, dtype=np.int64).reshape(-1)
    m      = int(len(states))

    # Map state labels to 0-based indices.
    # Caller passes states = [1, 2], so subtracting states[0] maps 1→0, 2→1.
    s_min = int(states[0])
    s0 = S_T[:-1] - s_min    # previous state, 0-indexed, shape (T-1,)
    s1 = S_T[1:]  - s_min    # next     state, 0-indexed, shape (T-1,)

    # Encode each (prev, next) pair as a scalar index: prev*m + next.
    # np.bincount then counts how many times each pair occurs — the
    # sufficient statistic for the Beta posterior on the transition
    # probabilities p = P(stay in recession) and q = P(stay in expansion).
    pairs  = s0 * m + s1                              # shape (T-1,)
    counts = np.bincount(pairs, minlength=m * m)      # shape (m*m,)

    return counts.reshape(m, m).astype(float)
