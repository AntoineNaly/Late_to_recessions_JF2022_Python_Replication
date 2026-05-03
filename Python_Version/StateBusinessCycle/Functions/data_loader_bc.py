# -*- coding: utf-8 -*-
"""
data_loader.py
--------------
Loads the raw macroeconomic and financial panel data from the .mat file,
standardises it, and constructs the quarter-end indicator used throughout
the state-space model.

Data used in estimation (Section III.A):
    y_monthly    — monthly macroeconomic indicators (N_m series, T observations)
    y_quarterly  — quarterly macroeconomic indicators (N_q series, T observations)
    NBER_rec_index — NBER recession dates (1=recession, 0=expansion), used to
                     initialise the regime path S_t for the Hamilton filter.

Standardisation:
    Each series is cross-sectionally standardised column-wise to have zero mean
    and unit standard deviation (NaN-safe, ddof=1 denominator):
        yy_{i,t} = (y_{i,t} - nanmean(y_i)) / nanstd(y_i)
    This makes the factor loadings γ_i in eq. (13) directly comparable across
    series with different natural units.

Quarter-end indicator (eq. 17 / IA.16):
    indexQuarter[t] = 1 at the last month of every quarter (months 3, 6, 9, ...).
    At these dates the Kalman filter uses the A_last selection matrix to include
    quarterly observables via the Mariano-Murasawa time aggregation (eq. 17).
    At all other months, only monthly observables enter through A_NotLast.
"""

from pathlib import Path
import numpy as np
from scipy.io import loadmat


def load_macro_data(data_path=None):
    """
    Load raw data from the .mat file, standardise, and build helper arrays.

    Parameters
    ----------
    data_path : str, Path, or None
        Explicit path to the .mat file.  If None, the function searches
        for 'Data/dataMacroFinance_1950_2019_updated.mat' relative to
        this source file's location (two levels up if inside a package).

    Returns
    -------
    yy_monthly    : (T, N_m) ndarray  -- standardised monthly data
    yy_quarterly  : (T, N_q) ndarray  -- standardised quarterly data
    NBER_rec_index : (T,) ndarray     -- 1=recession, 0=expansion
    indexQuarter  : (T,) int ndarray  -- 1 at every 3rd month (end of quarter)
    T             : int
    N_m           : int
    N_q           : int
    """

    # ------------------------------------------------------------------
    # Locate the .mat file
    # ------------------------------------------------------------------
    if data_path is None:
        try:
            here = Path(__file__).resolve().parent
        except NameError:
            here = Path.cwd()
        # Try both: sibling 'Data/' folder, or one level up
        candidates = [
            here / "Data" / "dataMacroFinance_1950_2019_updated.mat",
            here.parent / "Data" / "dataMacroFinance_1950_2019_updated.mat",
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break
        if data_path is None:
            raise FileNotFoundError(
                "Cannot find 'dataMacroFinance_1950_2019_updated.mat'. "
                "Pass data_path= explicitly or place the file in a 'Data/' "
                "folder next to data_loader.py."
            )

    # ------------------------------------------------------------------
    # Load .mat file.
    # Variable names in the file:
    #   y_monthly      -- raw monthly panel     (T, N_m)
    #   y_quarterly    -- raw quarterly panel   (T, N_q)
    #   NBER_rec_index -- NBER recession dummy  (T,)
    # ------------------------------------------------------------------
    mat = loadmat(str(data_path), simplify_cells=True)

    y_monthly   = np.array(mat["y_monthly"],   dtype=float)   # (T, N_m)
    y_quarterly = np.array(mat["y_quarterly"], dtype=float)   # (T, N_q)
    NBER_rec_index = np.array(mat["NBER_rec_index"], dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Standardise column-wise (NaN-safe, ddof=1 matching MATLAB nanstd).
    # Standardisation ensures that factor loadings γ_i in eq. (13) are
    # comparable across series and that the prior on γ_i (Table IV) is
    # scale-appropriate.
    # ------------------------------------------------------------------
    yy_monthly   = _standardise(y_monthly)
    yy_quarterly = _standardise(y_quarterly)

    T,   N_m = yy_monthly.shape
    _,   N_q = yy_quarterly.shape

    # ------------------------------------------------------------------
    # indexQuarter: 1 at the last month of every quarter (months 3, 6, 9, ...).
    # Used in generate_xt_sv to select the A_last measurement branch (eq. IA.13)
    # and to include quarterly observables via the aggregation function (eq. 17).
    # 0-indexed positions: [2, 5, 8, ...] = [2::3]
    # ------------------------------------------------------------------
    indexQuarter = np.zeros(T, dtype=int)
    indexQuarter[2::3] = 1

    return yy_monthly, yy_quarterly, NBER_rec_index, indexQuarter, T, N_m, N_q


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _standardise(y: np.ndarray) -> np.ndarray:
    """
    Column-wise NaN-safe standardisation: yy = (y - nanmean(y)) / nanstd(y)
    with ddof=1.  Columns that are all-NaN or constant are left unchanged
    (mu=0, sigma=1 applied).
    """
    mu    = np.nanmean(y, axis=0)   # (ncols,)
    sigma = np.nanstd(y,  axis=0, ddof=1)
    mu[np.isnan(mu)]       = 0.0
    sigma[(sigma == 0) | np.isnan(sigma)] = 1.0
    return (y - mu) / sigma
