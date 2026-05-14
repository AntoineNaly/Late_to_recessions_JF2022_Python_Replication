# -*- coding: utf-8 -*-

"""
main_BC.py
----------
Orchestrator for the Bayesian MCMC estimation of the state-space model in
Gómez-Cram (2022), "Late to Recessions: Stocks and the Business Cycle", JF.

The sampler iterates a Gibbs chain to draw from the joint posterior
p(Θ_{bc}, π_{1:T} | r^e_{1:T}, F_T) described in Section D (eq. 19).
Each iteration cycles through the following conditional posteriors:

    Step 1 — z_{1:T} | Θ_{bc}, S_{1:T}, data
        Kalman filter + smoother draw for the latent common growth factor
        path z_t (eq. 14) and all idiosyncratic states e_{i,t} (eq. 15),
        using the state-space representation (IA.13)–(IA.14).

    Step 2 — S_{1:T} | z_{1:T}, Θ_{bc}, data
        Hamilton filter (eq. 16) forward pass to compute filtered regime
        probabilities π_{t|t}, followed by a backward simulation draw of
        the full regime path S_{1:T}.

    Step 3 — (γ_i, ψ_i, σ²_{e,i}) | z_{1:T}, S_{1:T}, data   [monthly + quarterly]
        Gibbs sweep over all idiosyncratic parameters for each observable
        series in eqs. (13) and (15), using conjugate Normal/IG posteriors.

    Step 4 — (φ_z, μ_0, μ_1, σ²_{z,0}, h_z) | z_{1:T}, S_{1:T}, data
        GLS-based draws for the common factor parameters in eq. (14), with
        rejection sampling enforcing μ_1 > 0 (footnote 18).

    Step 5 — (p, q) | S_{1:T}
        Beta posterior draws for the Markov transition probabilities in the
        2-state chain governing S_t (eq. 14).

The first n0 iterations are discarded as burn-in; mm draws are retained.
Posterior summaries (median recession probability and common growth factor)
are computed and plotted after the chain completes.

Usage
-----
    python main_bc.py                      # full run (N0=15000, MM=25000)
    python main_bc.py --n0 100 --mm 200    # short test run
    python main_bc.py --seed 42

Or from Python:
    from main_bc import run_state_business_cycle
    results = run_state_business_cycle(seed=42, n0=100, mm=200)

Output dict keys
----------------
    'prob_rec'      : (T,) ndarray   — median posterior P(S_t=recession | data)
    'common_growth' : (T,) ndarray   — median posterior common growth factor z_t
    'Gamma_tot'     : (N_gamma, MM)  — all γ_i draws (post burn-in)
    'PSI_tot'       : (N_psi,   MM)  — all ψ_i draws
    'SIG_tot'       : (N_sig,   MM)  — all σ²_{e,i} draws
    'MU_G_tot'      : (2, MM)        — [μ_0, μ_1] draws
    'PHI_tot'       : (1, MM)        — φ_z draws
    'probState'     : (2, MM)        — [p, q] Markov transition probability draws
    'Prob_tot'      : (Tstar, 2, MM) — filtered regime probabilities π_{t|t}
    'State_common'  : (Tstar+1, 3, MM) — z_t path: [draw, filtered, predicted]
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add project root so sub-packages resolve correctly
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Output path — resolved automatically from the project structure.
#
# Expected layout (both folders must live under the same v0code parent):
#   v0code/
#     StateBusinessCycle/main_BC.py       ← this file
#     ExpectedReturnMeasure/Data/         ← output written here
#
# SCRIPT_DIR is v0code/StateBusinessCycle, so the sibling folder is
# SCRIPT_DIR.parent / "ExpectedReturnMeasure" / "Data".
# ---------------------------------------------------------------------------
_ER_DATA_DIR = SCRIPT_DIR.parent / "ExpectedReturnMeasure" / "Data"

# ---------------------------------------------------------------------------
# Imports — all from Functions sub-package
# ---------------------------------------------------------------------------
#from Functions.data_loader_bc           import load_macro_data
#from Functions.priors                    import build_priors
from Functions.specifyPriorsGibbsMacro  import specify_priors_gibbs_macro
from Functions.initialValuesMacro       import initial_values_macro
from Functions.generate_xt_sv           import generate_xt_sv
from Functions.hamiltonfilter_xt_sv     import hamiltonfilter_xt_sv
from Functions.gibbSamplingMacro        import gibbs_sampling_macro
from Functions.generate_MU_PHI_sv       import generate_mu_phi_sv
from Functions.generate_ChangeState     import generate_change_state
from Functions.utils                    import ensure_rng


# =============================================================================
# ── USER CONFIGURATION ────────────────────────────────────────────────────────
# All settings the user needs to change are here. No other edits required.
# =============================================================================

# requirements: pip install fredapi yfinance scipy openpyxl pandas numpy requests

FRED_API_KEY             = "YOUR_FRED_API_KEY"
# Free at: https://fred.stlouisfed.org/docs/api/api_key.html

MODE = "replicate"
#   "replicate" → replicates Gomez-Cram JF 2022 exactly
#                 data ends Dec 2019, FRED vintages pinned to March 2020
#   "latest"    → extends to most recent available data
#                 TEDRATE replaced by CP-Tbill (r=0.97, R²=0.93)

REPLICATE_VINTAGE_CUTOFF = "2020-03-31"
#   In replicate mode, current-vintage FRED series are pinned to the last
#   available vintage on or before REPLICATE_VINTAGE_CUTOFF. Each series has a different
#   release schedule (e.g. INDPRO → 2020-03-17, ICSA → 2020-03-26);
#   build_macro_dataset.py resolves the correct per-series date automatically.

# Data folder — auto-detected relative to this file. Contains:
#   hMvMd.xlsx                             Philly Fed aggregate weekly hours
#     → https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/
#       real-time-data/data-files/xlsx/hMvMd.xlsx
#   dataMacroFinance_1950_2019_updated.mat  author's original data (optional,
#                                           used only for correlation table)
_DATA_DIR = Path(__file__).resolve().parent / "Data"


ZSCORE_METHOD = "fullsample" # "fullsample" | "recursive"
# fullsample : author's approach — z-score over entire sample: hindsights bias, doesn't handle covid outlier well
# recursive  : expanding window — no look-ahead, real-time compatible, but assigns different values to same economic signal
ZSCORE_MIN_WINDOW = 120  # used if ZSCORE_METHOD = "recursive", minimum observations before recursive z-score kicks in, 
# early rows backfilled using first ZSCORE_MIN_WINDOW obs stats

# =============================================================================
# ── END USER CONFIGURATION ────────────────────────────────────────────────────
# =============================================================================


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def save_gibbs_output(results: dict, out_dir: Path,
                      bc_monthly_idx=None,
                      bc_nber_rec=None,
                      verbose: bool = True) -> Path:
    """
    Save the three quantities consumed by the ExpectedReturnMeasure code
    to a compressed numpy archive (.npz) in out_dir.

    Saved arrays (mirroring commonGrowthData_YYYY_YYYY.mat in MATLAB):
    -----------------------------------------------------------------------
    pi_t_mean         (Tstar,)  — posterior median of π_{t|t} = P(S_t=recession | F_t).
                                  Computed from Prob_tot[:, :, 0] (recession = state 0).
                                  Shape Tstar = T-3 (three obs lost to pre-whitening).

    commonGrowth_mean (Tstar,)  — posterior median of the common growth factor z_t.
                                  Taken from State_common[1:, 0, :] (draw column),
                                  dropping the prepended lag so the length matches
                                  pi_t_mean exactly (both Tstar).

    pi_ss             (2,)      — stationary distribution [π_rec, π_exp] of the
                                  Markov chain, computed from the posterior mean
                                  transition probabilities p̄ and q̄:
                                      π_rec = (1-q̄) / (2-p̄-q̄)
                                      π_exp = (1-p̄) / (2-p̄-q̄)
                                  Matches pi_ss = [0.1301; 0.8699] in MATLAB output.

    Loading in ExpectedReturnMeasure (one line):
        data = np.load(path_to_npz)
        pi_t_mean         = data["pi_t_mean"]
        commonGrowth_mean = data["commonGrowth_mean"]
        pi_ss             = data["pi_ss"]

    Parameters
    ----------
    results : dict   — return value of run_state_business_cycle
    out_dir : Path   — destination folder (created if absent)
    verbose : bool

    Returns
    -------
    Path to the saved .npz file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- pi_t_mean: posterior median of filtered recession probability π_{t|t} --
    # Prob_tot has shape (Tstar, 2, MM); column 0 = P(S_t=recession | F_t).
    # Median over the MM draw axis (axis=2) gives the point estimate used in eq. (6).
    Prob_arr  = results["Prob_tot"]                        # (Tstar, 2, MM)
    pi_t_mean = np.median(Prob_arr[:, 0, :], axis=1)      # (Tstar,)

    # -- commonGrowth_mean: posterior median of z_t (draw column) --
    # State_common has shape (Tstar+1, 3, MM); column 0 = Gibbs draw of z_t.
    # Row 0 is a prepended lag (from _reconstruct), so we take rows 1: to align
    # with pi_t_mean and obtain Tstar observations matching the MATLAB 660×1 output.
    SC_arr            = results["State_common"]            # (Tstar+1, 3, MM)
    commonGrowth_mean = np.median(SC_arr[1:, 0, :], axis=1)  # (Tstar,)

    # -- pi_ss: stationary distribution of the Markov chain --
    # probState has shape (2, MM): rows are [p, q] = [P(stay rec), P(stay exp)].
    # Posterior mean p̄, q̄ → stationary π via π_rec = (1-q̄)/(2-p̄-q̄).
    pq_arr = results["probState"]                          # (2, MM)
    p_bar  = float(pq_arr[0].mean())                      # posterior mean P(stay recession)
    q_bar  = float(pq_arr[1].mean())                      # posterior mean P(stay expansion)
    denom  = 2.0 - p_bar - q_bar
    pi_rec = (1.0 - q_bar) / denom                        # stationary P(recession)
    pi_exp = (1.0 - p_bar) / denom                        # stationary P(expansion)
    pi_ss  = np.array([pi_rec, pi_exp])                   # (2,) matches MATLAB [0.1301; 0.8699]

    out_path = out_dir / "commonGrowthData.npz"

    np.savez_compressed(
        out_path,
        pi_t_mean         = pi_t_mean,
        commonGrowth_mean = commonGrowth_mean,
        pi_ss             = pi_ss,
        monthly_idx       = np.array(bc_monthly_idx, dtype="datetime64[ns]")
                            if bc_monthly_idx is not None
                            else np.array([], dtype="datetime64[ns]"),
        nber_rec          = np.array(bc_nber_rec, dtype=np.int8)
                            if bc_nber_rec is not None
                            else np.array([], dtype=np.int8))

    if verbose:
        print(f"\nSaved Gibbs output → {out_path}")
        print(f"  pi_t_mean         shape={pi_t_mean.shape},  "
              f"range=[{pi_t_mean.min():.4f}, {pi_t_mean.max():.4f}]")
        print(f"  commonGrowth_mean shape={commonGrowth_mean.shape},  "
              f"range=[{commonGrowth_mean.min():.4f}, {commonGrowth_mean.max():.4f}]")
        print(f"  pi_ss             = [{pi_ss[0]:.4f}, {pi_ss[1]:.4f}]  "
              f"(sum={pi_ss.sum():.6f})")

    return out_path



# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_state_business_cycle(
    seed: int = 1234,
    n0:   int = 15_000,   # burn-in draws to discard
    mm:   int = 25_000,   # posterior draws to keep
    data_path=None,       # override .mat path (useful for testing)
    verbose: bool = True,
) -> dict:
    
    """
    Run the full Gibbs sampler for the business-cycle state-space model.

    Parameters
    ----------
    seed      : RNG seed for reproducibility.
    n0        : Number of burn-in draws discarded before posterior storage.
    mm        : Number of posterior draws retained after burn-in.
    data_path : Optional explicit path to the .mat data file.
    verbose   : Print iteration progress.

    Returns
    -------
    dict with posterior summaries and raw draws (see module docstring).
    """

    rng = ensure_rng(seed)

    # -----------------------------------------------------------------------
    # 1. Load and standardise data (Section III.A)
    # -----------------------------------------------------------------------
    _PKL_PATH = _DATA_DIR / f"GC_jf2022_dataset_{MODE}.pkl"
    if not _PKL_PATH.exists():
        if verbose:
            print("Dataset pkl not found — building dataset ...")
            print(f"  MODE={MODE}  DATA_DIR={_DATA_DIR}\n")
        from Functions.build_macro_dataset import build_dataset as _build_dataset
        _build_dataset(
            fred_api_key=FRED_API_KEY,
            data_dir=_DATA_DIR,
            mode=MODE,
            replicate_vintage_cutoff=REPLICATE_VINTAGE_CUTOFF,
            author_mat_file=str(_DATA_DIR / "dataMacroFinance_1950_2019_updated.mat"),
        )
    with open(_PKL_PATH, "rb") as f:
        _d = pickle.load(f)
    
    # as stated in JF paper, only keep macro variables, no financial variables, which deteriorate timeliness of recession probabilities:
    #_d.update({'yy_monthly': _d['yy_monthly'].iloc[:, :6], 'yy_monthly_raw': _d['yy_monthly_raw'].iloc[:, :6], 'N_m': 6})
    # deactivated since replication data package shared with JF paper includes financial variables as well (monthly variables 7 to 12)
    
    def _recursive_zscore(df: pd.DataFrame,
                          min_window: int = ZSCORE_MIN_WINDOW) -> np.ndarray:
        """
        Expanding window z-score: at each t, normalize using mean/std
        computed on all observations up to and including t.
        Real-time compatible — no look-ahead bias.
        Early rows (t < min_window) backfilled using first min_window obs stats
        to avoid NaN warm-up period breaking downstream OLS initialisations.
        """
        mu  = df.expanding(min_periods=min_window).mean()
        sig = df.expanding(min_periods=min_window).std()
        sig[sig == 0] = np.nan
        z = (df - mu) / sig
        init_mu  = df.iloc[:min_window].mean()
        init_sig = df.iloc[:min_window].std()
        init_sig[init_sig == 0] = np.nan
        z.iloc[:min_window] = (df.iloc[:min_window] - init_mu) / init_sig
        return z.values

    def _apply_zscore(df: pd.DataFrame) -> np.ndarray:
        """Apply z-score method selected by ZSCORE_METHOD config."""
        if ZSCORE_METHOD == "recursive":
            return _recursive_zscore(df)
        elif ZSCORE_METHOD == "fullsample":
            mu  = df.mean()
            sig = df.std()
            sig[sig == 0] = np.nan
            return ((df - mu) / sig).values
        else:
            raise ValueError(f"ZSCORE_METHOD must be 'recursive' or "
                             f"'fullsample', got '{ZSCORE_METHOD}'")

    NBER_rec_index = _d["nber_rec"].values
    indexQuarter   = _d["qtr_end"].values
    monthly_idx    = _d["monthly_idx"]
    T              = int(_d["T"])
    N_m            = int(_d["N_m"])
    N_q            = int(_d["N_q"])

    yy_monthly   = _apply_zscore(_d["yy_monthly_raw"]).copy()
    yy_quarterly = _apply_zscore(_d["yy_quarterly_raw"]).copy()

    if verbose:
        print(f"[zscore] Method='{ZSCORE_METHOD}'  T={T}  N_m={N_m}  N_q={N_q}")


    # -----------------------------------------------------------------------
    # COVID variance scaling — Holston, Laubach & Williams (2023) 
    # https://www.newyorkfed.org/research/staff_reports/sr1063
    # NY Fed Staff Report 1063, Table 1 (US estimates)
    #
    # Gomez-Cram (JF 2022) assumes i.i.d. Gaussian shocks with fixed R.
    # COVID violates this in two ways: (1) March-April 2020 drops are 10-20σ
    # events that pull regime parameters toward COVID magnitudes, making
    # pre-COVID recessions look mild; (2) the shutdown/reopening V-shape
    # creates mechanically negatively autocorrelated shocks, distorting the
    # Hamilton filter's transition probability estimates.
    #
    # Multiply measurement noise R by κ_t during the pandemic window.
    # Large κ_t → Kalman gain K_t → 0 → filter coasts on transition equation.
    # Implemented by scaling observations by 1/√κ_t, equivalent to κ_t×R.
    # κ values from HLW (2023) Table 1, applied uniformly across all series.
    #
    # Abandonned: even penalized strongly, 2-state state-space model cannot acomodate COVID
    # -----------------------------------------------------------------------
    if MODE == "latest":
        
        COVID_KAPPA_2020 = 9    # 2020-Q2 through 2020-Q4
        COVID_KAPPA_2021 = 1.8  # 2021
        COVID_KAPPA_2022 = 1.7  # 2022
        COVID_MONTHS_2020 = pd.date_range("2020-03-31", "2020-12-31", freq="ME")
        COVID_MONTHS_2021 = pd.date_range("2021-01-31", "2021-12-31", freq="ME")
        COVID_MONTHS_2022 = pd.date_range("2022-01-31", "2022-12-31", freq="ME")            
        
        for months, kappa in [
            (COVID_MONTHS_2020, COVID_KAPPA_2020),
            (COVID_MONTHS_2021, COVID_KAPPA_2021),
            (COVID_MONTHS_2022, COVID_KAPPA_2022),
        ]:
            mask         = np.isin(monthly_idx, months)
            scale_factor = 1.0 / np.sqrt(kappa)
            yy_monthly[mask,   :] *= scale_factor
            yy_quarterly[mask, :] *= scale_factor
    
        if verbose:
            print(f"[COVID R-inflation] HLW (2023) κ: "
                  f"2020={COVID_KAPPA_2020}  "
                  f"2021={COVID_KAPPA_2021}  "
                  f"2022={COVID_KAPPA_2022}")

    # -----------------------------------------------------------------------
    # 2. Prior hyperparameters (Table IV)
    # -----------------------------------------------------------------------
    #priors_macro_gibbs, markov_priors = build_priors()
    priors_macro_gibbs, markov_priors = specify_priors_gibbs_macro()

    # Priors for common factor parameters in eq. (14)
    R0_  = priors_macro_gibbs["R0_"]    # (1,1) prior precision for φ_z
    T0_  = priors_macro_gibbs["T0_"]    # (1,)  prior mean for φ_z
    R0_M = priors_macro_gibbs["R0_M"]   # (2,2) prior precision for [μ_0, μ_1]
    T0_M = priors_macro_gibbs["T0_M"]   # (2,)  prior mean for [μ_0, μ_1]
    D0_  = float(priors_macro_gibbs["D0_"])   # prior scale for variance (0 = diffuse)
    V0_  = float(priors_macro_gibbs["V0_"])   # prior d.o.f. for variance (0 = diffuse)

    # Beta prior pseudo-counts for Markov transition probabilities p and q
    U1_01_ = float(markov_priors["U1_01_"])
    U1_00_ = float(markov_priors["U1_00_"])
    U1_10_ = float(markov_priors["U1_10_"])

    # -----------------------------------------------------------------------
    # 3. OLS-based initial values for all parameters
    #    Also removes near-all-NaN columns and returns cleaned data with
    #    updated N_m, N_q (columns with >99% / >80% NaN are dropped).
    # -----------------------------------------------------------------------
    (param_macro_MH, param_macro_gibbs, s_t, yy_monthly, yy_quarterly, N_m, N_q) = initial_values_macro(
        yy_monthly,  yy_quarterly, NBER_rec_index,  markov_priors, rng)
    
    if verbose:
        print(f"After NaN-column removal: N_m={N_m}, N_q={N_q}")
        print(f"Starting Gibbs sampler: N0={n0} burn-in, MM={mm} draws "
              f"({n0+mm} total iterations)")
        from Functions.generate_xt_sv import _NUMBA_AVAILABLE
        print(f"Numba JIT active: {_NUMBA_AVAILABLE}", flush=True)

    # -----------------------------------------------------------------------
    # 4. Pre-allocate storage (only post-burn-in draws are kept).
    #    Arrays sizes depend on T and N_m/N_q which are only known after step 3,
    #    so lists are used and converted to arrays after the chain completes.
    # -----------------------------------------------------------------------
    CAPN = n0 + mm

    store_State_common  = []   # z_t path: draw, filtered, predicted (Tstar+1, 3)
    store_probState     = []   # Markov transition probs [p, q]
    store_Gamma_tot     = []   # factor loadings γ_i for all monthly+quarterly series
    store_PSI_tot       = []   # AR(1) coefficients ψ_i
    store_SIG_tot       = []   # idiosyncratic variances σ²_{e,i}
    store_MU_G_tot      = []   # regime means [μ_0, μ_1] from eq. (14)
    store_PHI_tot       = []   # AR(1) persistence φ_z from eq. (14)
    store_Prob_tot      = []   # filtered regime probabilities π_{t|t} (eq. 16)
    store_Sigma_X       = []   # [σ²_{z,0}, h_z] stochastic volatility parameters

    # -----------------------------------------------------------------------
    # 5. Gibbs sampler main loop — iterates the conditional posteriors
    #    described in eqs. (18)–(19) and Section D of the paper.
    # -----------------------------------------------------------------------
    # Warm-start: reuse the final filtered covariance P_{T|T} from the
    # previous Kalman pass as the initial covariance for the next iteration,
    # avoiding a Lyapunov solve on every call after the first.
    _Pt_kf = None

    _t_start = time.perf_counter()
    _t_last  = _t_start
    for indexSimul in range(1, CAPN + 1):

        # -------------------------------------------------------------------
        # Step 1: Draw the latent state path x_{1:T} | Θ_{bc}, S_{1:T}, data
        #
        # Runs the Kalman filter (eqs. IA.13–IA.14) forward in time and
        # draws one sample of the full state vector x_t = [z_t, z_{t-1}, ...,
        # e_{m1,t}, ..., e_{q1,t}, ...] from the filtered distribution
        # N(α̂_{t|t}, P_{t|t}) at each step.
        # The drawn z_t path (first column) feeds into all subsequent steps.
        # -------------------------------------------------------------------
        _ts = time.perf_counter()
        loglh, z_t, _Pt_kf = generate_xt_sv(
            yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs, indexQuarter, rng,
            Pt_prev=_Pt_kf)   # warm-start: None triggers Lyapunov solve on first iteration only
                                            
        x_t = z_t[:, 0]   # z_t draw column, shape (Tstar+1,)
        if verbose and indexSimul <= 5:
            print(f"    [timing iter {indexSimul}] generate_xt_sv:       {time.perf_counter()-_ts:.3f}s", flush=True)

        # -------------------------------------------------------------------
        # Step 2: Draw regime path S_{1:T} | z_{1:T}, Θ_{bc}, data
        #
        # Forward Hamilton filter computes the filtered probabilities
        # π_{t|t} = P(S_t | F_t) for all t (eq. 16), conditioning on the
        # drawn z_t path and the current common-factor parameters.
        # The backward pass then draws the full path S_{1:T} via Carter-Kohn.
        # STT is padded with three zeros at the front to align with the full
        # T-length time series (z_t has Tstar = T-3 observations).
        # -------------------------------------------------------------------
        _ts2 = time.perf_counter()
        S_T, FLT_PR = hamiltonfilter_xt_sv(x_t, param_macro_MH, rng)
        if verbose and indexSimul <= 5:
            print(f"    [timing iter {indexSimul}] hamiltonfilter:        {time.perf_counter()-_ts2:.3f}s", flush=True)
        STT = np.concatenate([[0, 0, 0], S_T])   # length T (= 3 + Tstar)
        s_t = STT

        # -------------------------------------------------------------------
        # Step 3a: Monthly Gibbs sweep
        #   Draws (γ_i, ψ_i, σ²_{e,i}) for each monthly series in eqs. (13)
        #   and (15), conditioning on the drawn z_t path.
        #   Data passed from t=2 onward to align with z_t length (Tstar+1).
        # -------------------------------------------------------------------
        param_macro_gibbs_aux_m = {
            "gamma_macro":   param_macro_gibbs["gamma_macro_m"],
            "psi_macro":     param_macro_gibbs["psi_macro_m"],
            "SIG2_i_macro":  param_macro_gibbs["SIG2_i_macro_m"],
        }
        _ts3 = time.perf_counter()
        gamma_m, psi_m, sig2_m = gibbs_sampling_macro(
            yy_monthly[2:, :],
            x_t,
            param_macro_gibbs_aux_m,
            priors_macro_gibbs,
            index_monthly=True,
            rng=rng,
        )
        if verbose and indexSimul <= 5:
            print(f"    [timing iter {indexSimul}] gibbs_macro_monthly:   {time.perf_counter()-_ts3:.3f}s", flush=True)
        param_macro_gibbs["gamma_macro_m"]  = gamma_m
        param_macro_gibbs["psi_macro_m"]    = psi_m
        param_macro_gibbs["SIG2_i_macro_m"] = sig2_m

        # -------------------------------------------------------------------
        # Step 3b: Quarterly Gibbs sweep
        #   Draws (γ_i, ψ_i, σ²_{e,i}) for each quarterly series in eqs. (13)
        #   and (15).  All quarterly variables use a scalar γ_i (no 4-lag path).
        #   Data passed from t=2 onward for the same alignment as monthly.
        # -------------------------------------------------------------------
        param_macro_gibbs_aux_q = {
            "gamma_macro":   param_macro_gibbs["gamma_macro_q"],
            "psi_macro":     param_macro_gibbs["psi_macro_q"],
            "SIG2_i_macro":  param_macro_gibbs["SIG2_i_macro_q"],
        }
        _ts4 = time.perf_counter()
        gamma_q, psi_q, sig2_q = gibbs_sampling_macro(
            yy_quarterly[2:, :],
            x_t,
            param_macro_gibbs_aux_q,
            priors_macro_gibbs,
            index_monthly=False,
            rng=rng,
        )
        if verbose and indexSimul <= 5:
            print(f"    [timing iter {indexSimul}] gibbs_macro_quarterly: {time.perf_counter()-_ts4:.3f}s", flush=True)
        param_macro_gibbs["gamma_macro_q"]  = gamma_q
        param_macro_gibbs["psi_macro_q"]    = psi_q
        param_macro_gibbs["SIG2_i_macro_q"] = sig2_q

        # Combined γ/ψ/σ² vectors across all series for storage
        gamma_macro  = np.concatenate([gamma_m, gamma_q])
        psi_macro    = np.concatenate([psi_m,   psi_q])
        SIG2_i_macro = np.concatenate([sig2_m,  sig2_q])

        # -------------------------------------------------------------------
        # Step 4: Draw common-factor parameters | z_{1:T}, S_{1:T}, data
        #   Draws φ_z, [μ_0, μ_1], σ²_{z,0}, h_z from eq. (14) via GLS
        #   posteriors.  STT[2:] aligns the regime path with the z_t draw
        #   (which starts at t=3 due to the pre-whitening lag).
        #   σ²_{z,0} and h_z are overridden to 1 and 0 after sampling
        #   (normalisation and equal-volatility restrictions).
        # -------------------------------------------------------------------
        _ts5 = time.perf_counter()
        phi_cc, paramMU, Sigma2_0_cc, h_cc = generate_mu_phi_sv(
            x_t,
            STT[2:],            # regime path aligned to z_t length
            param_macro_MH,
            R0_, T0_, R0_M, T0_M, D0_, V0_,
            rng,
        )
        if verbose and indexSimul <= 5:
            print(f"    [timing iter {indexSimul}] generate_mu_phi_sv:    {time.perf_counter()-_ts5:.3f}s", flush=True)
        param_macro_MH["paramMU"]      = paramMU
        param_macro_MH["Sigma2_0_cc"]  = Sigma2_0_cc
        param_macro_MH["h_cc"]         = h_cc
        param_macro_MH["phi_cc"]       = phi_cc

        # -------------------------------------------------------------------
        # Step 5: Draw Markov transition probabilities p and q | S_{1:T}
        #
        # The Beta posterior is conjugate to the Binomial likelihood for the
        # transition counts.  Given S_{1:T}, the sufficient statistics are
        # the observed transition counts n_{ij} from state i to state j:
        #   p = P(stay in recession) ~ Beta(n_{00} + U1_00_, n_{01} + U1_01_)
        #   q = P(stay in expansion) ~ Beta(n_{11} + U1_10_, n_{10} + U1_10_)
        # STT[4:T-2] trims the padded zeros from the front of STT and the
        # final two time points to match the MATLAB convention.
        # -------------------------------------------------------------------
        states_seq = (STT[4:T - 2] + 1).astype(int)   # states in {1,2}
        tranmat = generate_change_state(states_seq, states=[1, 2])
        # Posterior Beta parameters = prior pseudo-counts + observed transition counts
        A1TT = rng.beta(tranmat[0, 1] + U1_01_,  # n_{01} + prior: times we LEFT recession
                        tranmat[0, 0] + U1_00_)   # n_{00} + prior: times we STAYED in recession
        B1TT = rng.beta(tranmat[1, 0] + U1_10_,  # n_{10} + prior: times we LEFT expansion
                        tranmat[1, 1] + U1_10_)   # n_{11} + prior: times we STAYED in expansion

        q = 1.0 - B1TT    # P(stay in expansion)
        p = 1.0 - A1TT    # P(stay in recession)
        paramProb = np.array([A1TT, B1TT])
        param_macro_MH["paramProb"] = paramProb

        # -------------------------------------------------------------------
        # Step 6: Store draws (only after burn-in)
        # -------------------------------------------------------------------
        if indexSimul > n0:
            store_State_common.append(z_t)           # full latent factor path z_t
            store_probState.append(np.array([p, q])) # Markov transition probabilities
            store_Gamma_tot.append(gamma_macro)      # factor loadings γ_i for all series
            store_PSI_tot.append(psi_macro)          # AR(1) coefficients ψ_i
            store_SIG_tot.append(SIG2_i_macro)       # idiosyncratic variances σ²_{e,i}
            store_MU_G_tot.append(paramMU)           # regime-dependent means [μ_0, μ_1]
            store_PHI_tot.append(np.array([phi_cc])) # common factor AR persistence φ_z
            store_Prob_tot.append(FLT_PR)            # filtered regime probs π_{t|t} (eq. 16)
            store_Sigma_X.append(np.array([Sigma2_0_cc, h_cc]))  # [σ²_{z,0}, h_z]

        if verbose and indexSimul % 100 == 0:
            _t_now     = time.perf_counter()
            _sec_iter  = (_t_now - _t_last) / 100.0
            _eta_sec   = _sec_iter * (CAPN - indexSimul)
            _eta_h     = int(_eta_sec // 3600)
            _eta_m     = int((_eta_sec % 3600) // 60)
            _phase     = "burn-in" if indexSimul <= n0 else "draw   "
            print(
                f"  [{_phase}] iter {indexSimul:>6}/{CAPN}  "
                f"loglh={loglh:10.2f}  phi={phi_cc:.3f}  mu1={paramMU[1]:.4f}  "
                f"{_sec_iter:.2f}s/iter  ETA {_eta_h}h{_eta_m:02d}m",
                flush=True,
            )
            _t_last = _t_now

    # -----------------------------------------------------------------------
    # 6. Posterior summaries — compute median estimates π̂_{t|T} and ẑ_t
    # -----------------------------------------------------------------------

    # Stack draws: shape (mm, Tstar, 2) for Prob_tot, (mm, Tstar+1, 3) for State_common
    Prob_tot_arr    = np.stack(store_Prob_tot,   axis=0)   # (MM, Tstar, 2)
    State_common_arr = np.stack(store_State_common, axis=0) # (MM, Tstar+1, 3)

    # Median posterior recession probability and common growth factor over mm draws
    rec_prob_median   = np.median(Prob_tot_arr[:, :, 0], axis=0)      # (Tstar,): P(S_t=rec | data)
    common_gro_median = np.median(State_common_arr[:, :, 0], axis=0)  # (Tstar+1,): z_t draw median

    # Prepend NaNs to align with full T-length time axis:
    #   Recession prob: Tstar = T-3 (pre-whitening drops 3 obs), prepend 3 NaNs
    #   Common growth:  Tstar+1 = T-2, prepend 2 NaNs
    rec_prob_full  = np.concatenate([np.full(3, np.nan), rec_prob_median])
    common_gro_full = np.concatenate([np.full(2, np.nan), common_gro_median])


    # -----------------------------------------------------------------------
    # 7. Plots — median posterior recession probability (eq. 16) and z_t
    # -----------------------------------------------------------------------
    _dates = _d["monthly_idx"]   # DatetimeIndex, length T

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(_dates, rec_prob_full, color="tab:red", lw=1.2)
    axes[0].axhline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_title(r"Posterior recession probability $\mathrm{Pr}(s_t = 0 \mid \mathrm{data})$")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.4)
    # shade NBER recessions
    _rec = _d["nber_rec"].values.astype(bool)
    for _i in range(1, len(_rec)):
        if _rec[_i] and not _rec[_i-1]:
            _rs = _dates[_i]
        if not _rec[_i] and _rec[_i-1]:
            axes[0].axvspan(_rs, _dates[_i], color="gray", alpha=0.2)

    axes[1].plot(_dates, common_gro_full, color="tab:blue", lw=1.2)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_title("Common growth factor $z_t$ (posterior draw median)")
    axes[1].set_ylabel("Growth")
    axes[1].grid(True, alpha=0.4)
    for _i in range(1, len(_rec)):
        if _rec[_i] and not _rec[_i-1]:
            _rs = _dates[_i]
        if not _rec[_i] and _rec[_i-1]:
            axes[1].axvspan(_rs, _dates[_i], color="gray", alpha=0.2)

    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # 8. Print posterior percentiles (Table IV comparison)
    # -----------------------------------------------------------------------
    if verbose:
        PHI_arr = np.stack(store_PHI_tot,   axis=1)  # (1, MM)  — φ_z draws
        MU_arr  = np.stack(store_MU_G_tot,  axis=1)  # (2, MM)  — [μ_0^z, μ_1^z] draws
        pq_arr  = np.stack(store_probState, axis=1)  # (2, MM)  — [p, q] draws

        mu0z_pct = np.percentile(MU_arr[0],  [5, 50, 95])  # regime mean of z_t in recession
        mu1z_pct = np.percentile(MU_arr[1],  [5, 50, 95])  # regime mean increment (expansion − recession)
        phiz_pct = np.percentile(PHI_arr[0], [5, 50, 95])  # AR(1) persistence of z_t (eq. 14)
        p_pct    = np.percentile(pq_arr[0],  [5, 50, 95])  # P(stay in recession) = 1 - A1TT
        q_pct    = np.percentile(pq_arr[1],  [5, 50, 95])  # P(stay in expansion) = 1 - B1TT

        # These are the BC state-space model parameters (eq. 14).
        # They are NOT the Table IV parameters and do NOT appear in the README
        # comparison table.  Table IV parameters (mu_0, rho, corr_s, phi_rec,
        # phi_exp, sqrt_Var_rec, sqrt_Var_exp) are all estimated in main_ERF.py
        # and printed there under "Posterior distribution (post burn-in draws ...)".
        # In particular: mu_0^z here is the regime mean of the latent factor z_t
        # in standardised units (~-2.3 in recession), which is a completely
        # different quantity from the expected return model's mu_0 (~0.005).
        print(f"\n--- BC state-space model posterior "
              f"(post burn-in draws {n0+1:,}–{n0+mm:,}) ---")
        print("  [These are eq. (14) parameters, NOT the Table IV parameters]")
        print(f"  {'param':<26} {'5%':>10} {'50%':>10} {'95%':>10}")
        print(f"  {'mu_0^z (recession mean z_t)':<26} {mu0z_pct[0]:>10.4f} {mu0z_pct[1]:>10.4f} {mu0z_pct[2]:>10.4f}")
        print(f"  {'mu_1^z (expansion increment)':<26} {mu1z_pct[0]:>10.4f} {mu1z_pct[1]:>10.4f} {mu1z_pct[2]:>10.4f}")
        print(f"  {'phi_z (AR persistence z_t)':<26} {phiz_pct[0]:>10.4f} {phiz_pct[1]:>10.4f} {phiz_pct[2]:>10.4f}")
        print(f"  {'p (stay in recession)':<26} {p_pct[0]:>10.4f} {p_pct[1]:>10.4f} {p_pct[2]:>10.4f}")
        print(f"  {'q (stay in expansion)':<26} {q_pct[0]:>10.4f} {q_pct[1]:>10.4f} {q_pct[2]:>10.4f}")
        print("  --> Run main_ERF.py next to get Table IV posterior estimates.")

    if verbose:
        print("\n StateBusinessCycle completed.")

    # -----------------------------------------------------------------------
    # 9. Save output for ExpectedReturnMeasure
    #    Writes commonGrowthData.npz to v0code/ExpectedReturnMeasure/Data/.
    #    The three arrays saved here are loaded at the start of the
    #    expected-returns estimation to provide π_{t|t} and z_t.
    # -----------------------------------------------------------------------
    results_dict = dict(
        prob_rec       = rec_prob_full,
        common_growth  = common_gro_full,
        State_common   = np.stack(store_State_common, axis=2),  # (Tstar+1, 3, MM)
        probState      = np.stack(store_probState,    axis=1),  # (2, MM)
        Gamma_tot      = np.stack(store_Gamma_tot,    axis=1),
        PSI_tot        = np.stack(store_PSI_tot,      axis=1),
        SIG_tot        = np.stack(store_SIG_tot,      axis=1),
        MU_G_tot       = np.stack(store_MU_G_tot,     axis=1),  # (2, MM)
        PHI_tot        = np.stack(store_PHI_tot,      axis=1),  # (1, MM)
        Prob_tot       = np.stack(store_Prob_tot,     axis=2),  # (Tstar, 2, MM)
        Sigma_X        = np.stack(store_Sigma_X,      axis=1),  # (2, MM)
    )

    save_gibbs_output(results_dict, out_dir=_ER_DATA_DIR,
                          bc_monthly_idx = _d["monthly_idx"],
                          bc_nber_rec    = _d["nber_rec"].values,
                          verbose=verbose)

    return results_dict


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC Gibbs sampler")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n0",   type=int, default=15_000,
                        help="Burn-in draws (default 15000)")
    parser.add_argument("--mm",   type=int, default=25_000,
                        help="Posterior draws (default 25000)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .mat data file")
    args = parser.parse_args()

    run_state_business_cycle(
        seed=args.seed,
        n0=args.n0,
        mm=args.mm,
        data_path=args.data,
    )