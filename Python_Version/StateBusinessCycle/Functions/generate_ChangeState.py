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
    S_T = np.asarray(S_T).reshape(-1)
    states = np.asarray(states).reshape(-1)
    m = len(states)

    change_state = np.zeros((m, m), dtype=float)

    for t in range(1, len(S_T)):
        st_prev = int(S_T[t - 1])
        st_now = int(S_T[t])
        i = np.where(states == st_prev)[0][0]
        j = np.where(states == st_now)[0][0]
        change_state[i, j] += 1.0

    return change_state
