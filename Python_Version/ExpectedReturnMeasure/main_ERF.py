# -*- coding: utf-8 -*-

"""
main_ERF.py
-----------
Estimates the Expected Return Factor model of Gómez-Cram (2022),
"Late to Recessions: Stocks and the Business Cycle," Journal of Finance.

Implements the Metropolis-within-Gibbs sampler described in Appendix B.4:
  Step 2 — simulation smoother draw of μ_{1:T} (inside evalmod_mix)
  Step 3 — Random-Walk Metropolis-Hastings draw of Θ (this file)

The sampler targets p(Θ | μ_{1:T}, y_{1:T}) ∝ p(y|Θ) p(Θ)  (eq. 19),
where the likelihood is the mixture predictive density (eq. 5):
    p(r^e_{t+1} | Θ, r^e_{1:t}) =
        (1 − π̂_{t|t}) · p(r^e_{t+1} | Θ_Exp)
      +     π̂_{t|t}  · p(r^e_{t+1} | Θ_Rec)

Data inputs (no manual downloads required)
------------------------------------------
  commonGrowthData.npz   — output of main_BC.py (pi_t_mean, commonGrowth_mean,
                            pi_ss, monthly_idx, nber_rec)
  F-F_Research_Data_Factors_CSV.zip — downloaded automatically from Ken French
                            (Mkt-RF and RF monthly since July 1926)
  para.txt               — 7-element starting parameter vector in Data/
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
#from scipy.optimize import minimize

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# Expected layout:
#   v0code/
#     ExpectedReturnMeasure/
#       main_ERF.py
#       Functions/   ← all imports live here
#       Data/        ← commonGrowthData.npz, para.txt
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR / "Functions") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "Functions"))

from objfcnMixStates import objfcn_mix_states
from data_loader_er  import load_expected_return_data, _get_full_idx
from priors_er       import build_priors_and_bounds


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _parse_dates(date_data):
    """Convert date_data list to pandas DatetimeIndex, or None."""
    if date_data is None:
        return None
    try:
        return pd.to_datetime(date_data)
    except Exception:
        return None


def plot_figure4(dates, measure_expected_returns, ci_low, ci_high,
                 mean_excess_return, NBER_index):
    """
    Reproduce Figure 4 of Gomez-Cram (2022).

    Plots the filtered one-month-ahead excess return forecast μ̂_{t+1|t}
    (eq. 6, posterior median) with 90% credible interval, unconditional
    mean, and NBER recession shading.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    T = len(measure_expected_returns)
    x = dates if dates is not None else np.arange(T)

    # NBER recession shading
    NBER_index = np.asarray(NBER_index).reshape(-1)
    in_rec, rec_start = False, None
    for t in range(T):
        if NBER_index[t] == 1 and not in_rec:
            rec_start = x[t]; in_rec = True
        elif NBER_index[t] == 0 and in_rec:
            ax.axvspan(rec_start, x[t], color="lightgrey", alpha=0.6, lw=0)
            in_rec = False
    if in_rec:
        ax.axvspan(rec_start, x[-1], color="lightgrey", alpha=0.6, lw=0)

    ax.fill_between(x, ci_low, ci_high, color="steelblue", alpha=0.25,
                    label="90% credible interval")
    ax.plot(x, measure_expected_returns, color="steelblue", lw=1.2,
            label=r"One-month ahead excess return forecast $\hat{\mu}_{t+1|t}$")
    ax.axhline(mean_excess_return, color="black", lw=1.2,
               label="Mean excess return")

    ax.set_ylabel("Percentage")
    ax.set_title(
        r"Fig. 4 — One-month ahead excess return forecast $\hat{\mu}_{t+1|t}$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    if dates is not None:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
    else:
        ax.set_xlabel("Month (t)")

    plt.tight_layout()
    plt.show()


def plot_all_candidates(results, NBERIndex, burnin=0):
    """
    Plot all stored state series to identify which one matches Figure 4.
    Used as a diagnostic — compare visually with the paper.
    """
    candidates = {
        "X_pred_simul[:, 0, :]  (predicted state, μ_{t+1})":
            np.median(results["X_pred_simul"][:, 0, burnin:], axis=1) * 1200,
        "X_pred_simul[:, 1, :]  (predicted state, μ_t)":
            np.median(results["X_pred_simul"][:, 1, burnin:], axis=1) * 1200,
        "X_pred_simul[:, 2, :]  (predicted state, σ_r ε^r)":
            np.median(results["X_pred_simul"][:, 2, burnin:], axis=1) * 1200,
        "X_up_simul[:, 0, :]    (filtered state, μ_{t+1})":
            np.median(results["X_up_simul"][:, 0, burnin:], axis=1) * 1200,
        "X_up_simul[:, 1, :]    (filtered state, μ_t)":
            np.median(results["X_up_simul"][:, 1, burnin:], axis=1) * 1200,
        "X_sm_simul[:, 0, :]    (smoothed draw, μ_{t+1})":
            np.median(results["X_sm_simul"][:, 0, burnin:], axis=1) * 1200,
        "X_up_mix               (mixture filtered μ_t)":
            np.median(results["X_up_mix"][:, burnin:], axis=1) * 1200,
    }
    T = len(NBERIndex)
    fig, axes = plt.subplots(len(candidates), 1,
                             figsize=(12, 3 * len(candidates)), sharex=True)
    for ax, (label, series) in zip(axes, candidates.items()):
        in_rec, rec_start = False, None
        for t in range(T):
            if NBERIndex[t] == 1 and not in_rec:
                rec_start = t; in_rec = True
            elif NBERIndex[t] == 0 and in_rec:
                ax.axvspan(rec_start, t, color="lightgrey", alpha=0.6, lw=0)
                in_rec = False
        if in_rec:
            ax.axvspan(rec_start, T, color="lightgrey", alpha=0.6, lw=0)
        ax.plot(np.arange(T), series, lw=1, color="steelblue")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Filter sign diagnostic
# ---------------------------------------------------------------------------

def test_filter_sign():
    """
    Verify that a large negative return produces a negative update to μ_{t+1}
    under the recession parameters from para.txt.
    """
    from evalmod import evalmod
    sigma2_rec = 0.0015 * (1 + 1.3842)
    para_rec   = np.array([0.0065, 0.97, -0.955, 0.145, sigma2_rec])
    YY_test    = np.array([[0.01], [-0.30], [0.01]])
    _, _, At_mat, Kg_mat, At_pred = evalmod(
        para_rec, YY_test, indexMinimize=1, rng=np.random.default_rng(0)
    )
    print(f"K[0] at crash period:       {Kg_mat[1, 0]:.6f}  (expect > 0)")
    print(f"At_pred[2, 0] after crash:  {At_pred[2, 0]:.6f}  (expect < 0)")
    print(f"Annualised:                 {At_pred[2, 0]*1200:.2f}%  (expect negative)")


# ---------------------------------------------------------------------------
# Main estimation routine
# ---------------------------------------------------------------------------

def main_erf(
    nsim=25_000,
    burnin=15_000,
    cc0=0.0001,
    cc=0.00025,
    seed=1234,
    plot_results=True,
):
    """
    Run the Metropolis-within-Gibbs sampler for the expected return model.

    Parameters
    ----------
    nsim         : MH draws to store (paper uses 25,000)
    burnin       : draws to discard post-hoc (paper uses 15,000)
    cc0          : tight proposal scaling for the initialisation search
    cc           : proposal scaling for main MH loop (target ~30% acceptance)
    seed         : random seed for reproducibility
    plot_results : if True, produces Figure 4 after the sampler

    Returns
    -------
    results : dict — parameter draws, state draws, likelihood traces,
                     posterior-median expected return series
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Data
    #    Sample window is determined dynamically from commonGrowthData.npz:
    #      ER_START = Jan 1965 (fixed)
    #      ER_END   = end date of main_BC sample (Dec 2019 in replicate mode,
    #                 latest quarter end in latest mode)
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    YY, rf, NBERIndex, pi_t, z_t, sigma2_1_fix = load_expected_return_data(
        base_dir)

    # Replace any NaNs in pi_t with 0, matching MATLAB: pi_t(isnan(pi_t))=0
    pi_t = np.where(np.isnan(pi_t), 0.0, pi_t)

    T        = YY.shape[0]
    n_states = 3
    npara    = 7

    # Build date axis for plots from the ER sample index
    _full_idx = _get_full_idx()
    dates     = _full_idx if len(_full_idx) == T else None

    print(f"Data loaded: T={T}  "
          f"({_full_idx[0].strftime('%Y-%m') if dates is not None else '?'} → "
          f"{_full_idx[-1].strftime('%Y-%m') if dates is not None else '?'})")
    print(f"sigma2_1_fix={sigma2_1_fix:.8f}")

    # ------------------------------------------------------------------
    # 2. Priors, bounds, and proposal covariance
    # ------------------------------------------------------------------
    (
        pshape, pmean, pstdd,
        pmask, pmaskinv, pfix,
        lubound, sigscale,
    ) = build_priors_and_bounds(YY, sigma2_1_fix)

    # ------------------------------------------------------------------
    # 3. Starting parameters from para.txt
    # ------------------------------------------------------------------
    para_txt_path = base_dir / "Data" / "para.txt"
    para_old = np.loadtxt(para_txt_path).astype(float)
    para_old = para_old * pmaskinv + pfix * pmask
    print(f"Loaded starting parameters from para.txt: {para_old}")

    indexMinimize = 0   # 0 = return log posterior (maximise)

    def fcn_theta(theta):
        return objfcn_mix_states(
            theta, YY, pi_t, indexMinimize,
            pshape, pmean, pstdd,
            pmask, pmaskinv, pfix, lubound,
            rng=rng,
        )

    # ------------------------------------------------------------------
    # 4. Initialisation — perturb from starting point
    # ------------------------------------------------------------------
    (
        post_old, like_old,
        At_draw_tot_old, At_mat_tot_old, At_pred_tot_old,
        Kgain_old, loglh_tot_old,
        modelInfo_1_old, modelInfo_2_old,
    ) = fcn_theta(para_old)

    found = False
    while not found:
        jump = rng.multivariate_normal(mean=para_old, cov=cc0 * sigscale)
        jump = jump * pmaskinv + pfix * pmask
        (
            post_tmp, like_tmp,
            At_draw_tmp, At_mat_tmp, At_pred_tmp,
            Kgain_tmp, loglh_tot_tmp,
            modelInfo_1_tmp, modelInfo_2_tmp,
        ) = fcn_theta(jump)
        if post_tmp > -1e6:
            para_old        = jump
            post_old        = post_tmp
            like_old        = like_tmp
            At_draw_tot_old = At_draw_tmp
            At_mat_tot_old  = At_mat_tmp
            At_pred_tot_old = At_pred_tmp
            Kgain_old       = Kgain_tmp
            loglh_tot_old   = loglh_tot_tmp
            modelInfo_1_old = modelInfo_1_tmp
            modelInfo_2_old = modelInfo_2_tmp
            found           = True

    # ------------------------------------------------------------------
    # 5. Storage allocation
    # ------------------------------------------------------------------
    parasim      = np.zeros((nsim, npara))
    likesim      = np.zeros(nsim)
    postsim      = np.zeros(nsim)
    rej          = np.zeros(nsim)

    X_sm_simul   = np.zeros((T, n_states, nsim))
    X_up_simul   = np.zeros((T, n_states, nsim))
    X_pred_simul = np.zeros((T, n_states, nsim))

    kg_sim       = np.zeros((nsim, 2))
    logLikiMix   = np.zeros((T, nsim))
    logLikiMod_1 = np.zeros((T, nsim))
    logLikiMod_2 = np.zeros((T, nsim))
    X_up_mix     = np.zeros((T, nsim))
    X_up_Mod_1   = np.zeros((T, nsim))
    X_up_Mod_2   = np.zeros((T, nsim))

    lbd_low  = lubound[:, 0]
    lbd_high = lubound[:, 1]

    # Pre-compute Cholesky of proposal covariance c·Ω once
    L_prop = np.linalg.cholesky(cc * sigscale)

    # ------------------------------------------------------------------
    # 6. Random-Walk Metropolis-Hastings  (Appendix B.4, step 3)
    # ------------------------------------------------------------------
    n_accept = 0

    with tqdm(
        total=nsim,
        desc="MH Sampler",
        unit="draw",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
        ),
    ) as pbar:

        for idx in range(nsim):

            # Propose Θ_new within admissible bounds
            in_bounds = False
            while not in_bounds:
                theta_new = para_old + L_prop @ rng.standard_normal(npara)
                theta_new = theta_new * pmaskinv + pfix * pmask
                in_bounds = bool(
                    np.all((theta_new > lbd_low) & (theta_new < lbd_high))
                )

            # Evaluate posterior at proposal
            (
                post_new, like_new,
                At_draw_tot_new, At_mat_tot_new, At_pred_tot_new,
                Kgain_new, loglh_tot_new,
                modelInfo_1_new, modelInfo_2_new,
            ) = fcn_theta(theta_new)

            # MH accept / reject
            log_alpha   = post_new - post_old
            accept_prob = min(1.0, np.exp(log_alpha))

            if rng.uniform() <= accept_prob:
                para_old        = theta_new
                post_old        = post_new
                like_old        = like_new
                At_draw_tot_old = At_draw_tot_new
                At_mat_tot_old  = At_mat_tot_new
                At_pred_tot_old = At_pred_tot_new
                Kgain_old       = Kgain_new
                loglh_tot_old   = loglh_tot_new
                modelInfo_1_old = modelInfo_1_new
                modelInfo_2_old = modelInfo_2_new
                n_accept        += 1
            else:
                rej[idx] = 1

            # Store chain state
            parasim[idx, :]         = para_old
            likesim[idx]            = like_old
            postsim[idx]            = post_old
            X_sm_simul[:, :, idx]   = At_draw_tot_old
            X_up_simul[:, :, idx]   = At_mat_tot_old
            X_pred_simul[:, :, idx] = At_pred_tot_old
            kg_sim[idx, :]          = Kgain_old
            logLikiMix[:, idx]      = loglh_tot_old
            logLikiMod_1[:, idx]    = modelInfo_1_old[:, 0]
            logLikiMod_2[:, idx]    = modelInfo_2_old[:, 0]
            X_up_mix[:, idx]        = At_mat_tot_old[:, 1]
            X_up_Mod_1[:, idx]      = modelInfo_1_old[:, 2]
            X_up_Mod_2[:, idx]      = modelInfo_2_old[:, 2]

            if (idx + 1) % 100 == 0:
                pbar.set_postfix(
                    accept=f"{n_accept / (idx + 1):.1%}",
                    logpost=f"{post_old:.1f}",
                    refresh=False,
                )
                pbar.update(100)

        if nsim % 100:
            pbar.update(nsim % 100)

    print(f"\nFinal acceptance rate: {n_accept / nsim:.1%}  "
          f"(target ≈ 30%; adjust `cc` if far off)")

    # ------------------------------------------------------------------
    # 7. Posterior expected return series  (eq. 6)
    # ------------------------------------------------------------------
    post_draws = X_pred_simul[:, 0, burnin:]

    measure_expected_returns = np.median(post_draws, axis=1) * 1200.0
    ci_low  = np.percentile(post_draws,  5, axis=1) * 1200.0
    ci_high = np.percentile(post_draws, 95, axis=1) * 1200.0
    mean_excess_return = float(np.mean(YY[:, 0]) * 1200.0)

    if plot_results:
        plot_figure4(
            dates=dates,
            measure_expected_returns=measure_expected_returns,
            ci_low=ci_low,
            ci_high=ci_high,
            mean_excess_return=mean_excess_return,
            NBER_index=NBERIndex,
        )

    return dict(
        parasim=parasim,
        postsim=postsim,
        likesim=likesim,
        rej=rej,
        burnin=burnin,
        X_sm_simul=X_sm_simul,
        X_up_simul=X_up_simul,
        X_pred_simul=X_pred_simul,
        kg_sim=kg_sim,
        logLikiMix=logLikiMix,
        logLikiMod_1=logLikiMod_1,
        logLikiMod_2=logLikiMod_2,
        X_up_mix=X_up_mix,
        X_up_Mod_1=X_up_Mod_1,
        X_up_Mod_2=X_up_Mod_2,
        measure_expected_returns=measure_expected_returns,
        ci_low=ci_low,
        ci_high=ci_high,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    test_filter_sign()

    results = main_erf(nsim=25_000, burnin=15_000, plot_results=True)

    _, _, NBERIndex, _, _, _ = load_expected_return_data()

    plot_all_candidates(results, NBERIndex, burnin=0)

    # --- Diagnostics ---
    print("\n=== Likelihood / filter sanity checks ===")
    loglh       = results["logLikiMix"]
    total_loglh = loglh.sum(axis=0)
    print(f"Mean total log-likelihood : {total_loglh.mean():.2f}")
    print(f"Std                       : {total_loglh.std():.2f}")
    print(f"Min / Max                 : {total_loglh.min():.2f} / "
          f"{total_loglh.max():.2f}")

    loglh_med = np.median(loglh, axis=1)
    print(f"\nPer-period loglh (median): mean={loglh_med.mean():.3f}  "
          f"std={loglh_med.std():.3f}  min={loglh_med.min():.3f}  "
          f"max={loglh_med.max():.3f}")
    print(f"NaNs: {np.isnan(loglh_med).sum()}   "
          f"Infs: {np.isinf(loglh_med).sum()}")
    print(f"\nRejection rate: {results['rej'].mean():.1%}  "
          f"(acceptance={1-results['rej'].mean():.1%}, target≈30%)")

    parasim      = results["parasim"]
    burnin_diag  = results["burnin"]
    parasim_post = parasim[burnin_diag:, :]

    # Transform h and sigma2_1 to paper-comparable units (Table IV):
    #   √Var(r^e | S_t=Exp.) = √sigma2_1            (col 6)
    #   √Var(r^e | S_t=Rec.) = h × √sigma2_1        (col 5 × √col 6)
    sqrt_sigma2_1 = np.sqrt(parasim_post[:, 6])         # (MM,)
    sqrt_var_rec  = parasim_post[:, 5] * sqrt_sigma2_1  # (MM,)

    # Assemble paper-comparable draw matrix (MM, 7)
    parasim_print = np.column_stack([
        parasim_post[:, 0],   # mu_0
        parasim_post[:, 1],   # rho
        parasim_post[:, 2],   # corr_s
        parasim_post[:, 4],   # phi_2  = phi(S_t=Rec.)
        parasim_post[:, 3],   # phi_1  = phi(S_t=Exp.)
        sqrt_var_rec,          # sqrt_Var(Rec.) = h * sqrt(sigma2_1)
        sqrt_sigma2_1,         # sqrt_Var(Exp.) = sqrt(sigma2_1)
    ])

    param_names = [
        "mu_0", "rho", "corr_s",
        "phi(Exp.)", "phi(Rec.)",
        "sqrt_Var(Rec.)", "sqrt_Var(Exp.)",
    ]
    table_iv = {
        "mu_0":           (0.0053, 0.0065, 0.0076),
        "rho":            (0.958,  0.970,  0.982),
        "corr_s":         (-0.984, -0.955, -0.926),
        "phi(Rec.)":      (0.093,  0.149,  0.181),
        "phi(Exp.)":      (0.006,  0.009,  0.015),
        "sqrt_Var(Rec.)": (0.043,  0.060,  0.072),
        "sqrt_Var(Exp.)": (0.0389, 0.0389, 0.0389),
    }
    print(f"\nPosterior distribution "
          f"(post burn-in draws {burnin_diag}–{len(parasim)}):")
    print(f"  {'param':16s}  {'5%':>10}  {'50%':>10}  {'95%':>10}   "
          f"(Table IV ref)")
    for name, col in zip(param_names, parasim_print.T):
        p5, p50, p95 = np.percentile(col, [5, 50, 95])
        ref = table_iv.get(name)
        ref_str = f"  ref: {ref}" if ref else ""
        print(f"  {name:16s}  {p5:10.6f}  {p50:10.6f}  {p95:10.6f}{ref_str}")

    # Log-likelihood and parameter traces
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(total_loglh, lw=0.8, color="steelblue")
    axes[0].set_title("Total log-likelihood trace")
    axes[0].set_xlabel("MH draw")
    axes[0].set_ylabel("log p(y|Θ)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(parasim[:, 1], lw=0.8, color="darkorange", label="rho")
    axes[1].plot(parasim[:, 2], lw=0.8, color="steelblue",  label="corr_s")
    axes[1].set_title("Parameter traces: rho and corr_s")
    axes[1].set_xlabel("MH draw")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()