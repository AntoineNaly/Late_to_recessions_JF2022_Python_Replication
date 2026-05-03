# -*- coding: utf-8 -*-

"""
gibbSamplingMacro.py
--------------------
One Gibbs sweep over all idiosyncratic parameters for a single frequency
block (monthly or quarterly), updating ψ_i, σ²_{e,i}, and γ_i in turn.

The idiosyncratic component e_{i,t} follows an AR(1) process (eq. 15):
    e_{i,t+1} = μ_i + ψ_i e_{i,t} + σ_i ε_{i,t+1},   ε_{i,t+1} ~ i.i.d. N(0,1)

Each observable y_{i,t} loads on the common growth factor z_t via γ_i (eq. 13):
    y_{i,t+1} = γ_i z_{t+1} + e_{i,t+1}

The Gibbs sweep proceeds in two passes:
    1. Draw (ψ_i, σ²_{e,i}) for each variable i, conditioning on γ_i and
       the current state-path draw z_t from generate_xt_sv.
    2. Draw γ_i for each variable i, conditioning on the just-drawn (ψ_i, σ²_{e,i}).

Both draws use Gaussian conjugate posteriors derived from the pre-whitened
regression implied by equations (13) and (15); see generate_PSIandSIG_macro
and generate_gamma_macro for the analytic posterior formulae.

Called twice per Gibbs iteration (once for monthly, once for quarterly).

indexMonthly flag
-----------------
True  (monthly):
  - The LAST variable triggers a 4-lag special case: its gamma vector has
    4 entries (loaded on z_t, z_{t-1}, z_{t-2}, z_{t-3}) rather than a scalar.
  - gamma_macro has length N_m+3.

False (quarterly):
  - All variables use a scalar gamma.  The 4-lag path is never triggered.
  - gamma_macro has length N_q.

gamma_macro accumulation
------------------------
The output gamma_macro is rebuilt by concatenating per-variable draws:
  - Non-last monthly or any quarterly variable: contributes 1 value.
  - Last monthly variable: contributes 4 values.
  Result for monthly:   length N_m-1 + 4 = N_m+3
  Result for quarterly: length N_q
"""

import numpy as np

try:
    from .generate_PSIandSIG_macro import generate_psi_and_sig_macro
    from .generate_gamma_macro     import generate_gamma_macro
except ImportError:
    from generate_PSIandSIG_macro import generate_psi_and_sig_macro
    from generate_gamma_macro     import generate_gamma_macro


def gibbs_sampling_macro(
    yy,
    x_t,
    param_macro_gibbs_aux,
    priors_macro_gibbs,
    index_monthly,
    rng,
):
    """
    Parameters
    ----------
    yy : (T_slice, nVars) ndarray
        Data for this frequency (NaNs permitted; they are masked per-variable).
        Caller passes yy_monthly[2:, :] or yy_quarterly[2:, :].
    x_t : (T_slice,) ndarray
        Common growth factor z_t, same length as yy.
    param_macro_gibbs_aux : dict
        'gamma_macro'   : (N+3,) or (N,) ndarray  -- current γ draws
        'psi_macro'     : (N,) ndarray             -- current ψ draws
        'SIG2_i_macro'  : (N,) ndarray             -- current σ² draws
    priors_macro_gibbs : dict
        'R0_V'  : (1,1) ndarray or float  -- prior precision for ψ  (eye(1)/4)
        'T0_V'  : (1,) array or float     -- prior mean for ψ       ([0])
        'V0_'   : float                   -- prior d.o.f.             (0)
        'D0_'   : float                   -- prior scale              (0)
        'R00_'  : float or (1,1)          -- prior precision, scalar γ (1/4)
        'T00_'  : float or (1,)           -- prior mean, scalar γ     (0)
        'R00_4' : (4,4) ndarray           -- prior precision, 4-lag γ
        'T00_4' : (4,) ndarray            -- prior mean, 4-lag γ
    index_monthly : bool
        True for monthly block, False for quarterly.
    rng : np.random.Generator

    Returns
    -------
    gamma_macro  : ndarray  -- updated gamma vector
    psi_macro    : (nVars,) ndarray
    SIG2_i_macro : (nVars,) ndarray
    """

    yy  = np.asarray(yy,  dtype=float)
    x_t = np.asarray(x_t, dtype=float).reshape(-1)   # z_t path drawn from generate_xt_sv
    nVars = yy.shape[1]

    # Unpack current parameter draws (copies to avoid aliasing)
    gamma_macro   = np.asarray(param_macro_gibbs_aux["gamma_macro"],   dtype=float).copy()  # current γ_i draws
    SIG2_i_macro  = np.asarray(param_macro_gibbs_aux["SIG2_i_macro"],  dtype=float).copy()  # current σ²_{e,i} draws
    psi_macro     = np.asarray(param_macro_gibbs_aux["psi_macro"],     dtype=float).copy()  # current ψ_i draws

    # Unpack priors (see specifyPriorsGibbsMacro)
    R0_V  = priors_macro_gibbs["R0_V"]          # prior precision for ψ_i — eye(1)/4
    T0_V  = priors_macro_gibbs["T0_V"]          # prior mean for ψ_i — 0
    V0_   = float(priors_macro_gibbs["V0_"])    # prior d.o.f. for σ²_{e,i} — 0
    D0_   = float(priors_macro_gibbs["D0_"])    # prior scale for σ²_{e,i} — 0
    R00_  = priors_macro_gibbs["R00_"]          # prior precision for scalar γ_i — 1/4
    T00_  = priors_macro_gibbs["T00_"]          # prior mean for scalar γ_i — 0
    R00_4 = priors_macro_gibbs["R00_4"]         # prior precision for 4-lag γ_i — (4,4)
    T00_4 = priors_macro_gibbs["T00_4"]         # prior mean for 4-lag γ_i — (4,)

    # ------------------------------------------------------------------
    # Step 1: Draw (ψ_i, σ²_{e,i}) for each variable
    # Conditions on current γ_i and the drawn z_t path.
    # Posterior is derived from the pre-whitened regression of eq. (IA.15).
    # ------------------------------------------------------------------
    for selectVar_0 in range(nVars):     # loop over variables
        selectVar = selectVar_0 + 1      # 1-indexed for sub-functions

        # NaN-safe data selection for this variable
        yy_col  = yy[:, selectVar_0]
        mask    = ~np.isnan(yy_col)
        yy_sel  = yy_col[mask]
        xt_sel  = x_t[mask]

        SIG2_i = float(SIG2_i_macro[selectVar_0])

        # Extract γ_i for this variable
        if index_monthly and selectVar == nVars:
            # Last monthly variable: 4-lag γ vector
            gamma_i = gamma_macro[nVars - 1:]   # shape (4,)
            sv_call = selectVar                 # triggers 4-lag case in sub-function
        else:
            # Scalar γ_i for all other variables
            gamma_i = float(gamma_macro[selectVar_0])
            sv_call = 1 if not index_monthly else selectVar
        # For quarterly, always pass sv_call=1 so the 4-lag path is never triggered
        nv_call = nVars if index_monthly else nVars + 1

        psi_new, sig2_new = generate_psi_and_sig_macro(
            yy_sel, xt_sel, gamma_i, SIG2_i,
            R0_V, T0_V, V0_, D0_,
            sv_call, nv_call,
            rng,
        )
        psi_macro[selectVar_0]    = psi_new
        SIG2_i_macro[selectVar_0] = sig2_new

    # ------------------------------------------------------------------
    # Step 2: Draw γ_i for each variable
    # Conditions on the just-drawn (ψ_i, σ²_{e,i}) and the z_t path.
    # Posterior is derived from the pre-whitened regression of eq. (IA.15).
    # ------------------------------------------------------------------
    gamma_macro_0 = []   # accumulated new gamma vector

    for selectVar_0 in range(nVars):
        selectVar = selectVar_0 + 1

        # NaN-safe data selection
        yy_col = yy[:, selectVar_0]
        mask   = ~np.isnan(yy_col)
        yy_sel = yy_col[mask]
        xt_sel = x_t[mask]

        SIG2_i = float(SIG2_i_macro[selectVar_0])
        PSI_i  = float(psi_macro[selectVar_0])

        # Quarterly branch: always scalar γ, so pass sv_call=1
        sv_call = 1 if not index_monthly else selectVar
        nv_call = nVars if index_monthly else nVars + 1

        gamma_temp = generate_gamma_macro(
            yy_sel, xt_sel, PSI_i, SIG2_i,
            R00_, T00_, R00_4, T00_4,
            sv_call, nv_call,
            rng,
        )
        gamma_macro_0.append(gamma_temp)

    # Concatenate: [scalar, ..., scalar, (4,)] for monthly  → (N_m+3,)
    #              [scalar, ..., scalar]        for quarterly → (N_q,)
    gamma_macro = np.concatenate(gamma_macro_0)

    return gamma_macro, psi_macro, SIG2_i_macro
