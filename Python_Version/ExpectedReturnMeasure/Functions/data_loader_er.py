# -*- coding: utf-8 -*-
"""
data_loader_er.py
-----------------
Loads all inputs for the Expected Return Factor estimation (main_ERF.py).

No FRED API key required — all data comes from the npz and the FF zip.
  Run main_BC.py first to generate commonGrowthData.npz.

Data sources
------------
commonGrowthData.npz   — output of main_BC.py:
                            pi_t_mean, pi_ss, commonGrowth_mean,
                            monthly_idx, nber_rec

F-F_Research_Data_Factors_CSV.zip       — Ken French: Mkt-RF, SMB, HML, RF (monthly)
F-F_Research_Data_Factors_daily_CSV.zip — Ken French: Mkt-RF daily (realized variance)

F-F_Momentum_Factor_CSV.zip        — Ken French: Mom (monthly)
Shiller ie_data.xls                — Robert Shiller: CAPE ratio (monthly since 1871)

Path resolution
---------------
  this file : .../ExpectedReturnMeasure/Functions/data_loader_er.py
  data dir  : .../ExpectedReturnMeasure/Data/
"""

from __future__ import annotations
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
# ER sample starts at the paper's fixed start date (Jan 1965).
# ER_END is determined dynamically from monthly_idx in commonGrowthData.npz.
ER_START = "1965-01-01"
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
_FF_URL   = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
             "ftp/F-F_Research_Data_Factors_CSV.zip")
_MOM_URL      = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                 "ftp/F-F_Momentum_Factor_CSV.zip")
_FF_DAILY_URL  = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                  "ftp/F-F_Research_Data_Factors_daily_CSV.zip")
_SHILLER_URL   = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
                  "downloads/7fd201b2-28ad-476c-bc67-7a2cab5304a3/ie_data.xls")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_full_idx() -> pd.DatetimeIndex:
    """
    Build the ER sample DatetimeIndex from ER_START to the end date stored
    in commonGrowthData.npz. Falls back to Dec 2019 if npz not found.
    """
    npz_path = _DATA_DIR / "commonGrowthData.npz"
    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        if "monthly_idx" in data and len(data["monthly_idx"]) > 0:
            bc_idx = pd.DatetimeIndex(data["monthly_idx"])
            er_end = bc_idx.max().strftime("%Y-%m-%d")
            return pd.date_range(ER_START, er_end, freq="ME")
    return pd.date_range(ER_START, "2019-12-31", freq="ME")


def _squeeze(mat_dict: dict, key: str) -> np.ndarray:
    return np.squeeze(np.asarray(mat_dict[key], dtype=float))


def _load_ff_zip(url: str, col_names: list) -> pd.DataFrame:
    """
    Download a Ken French CSV zip and return monthly data as a DataFrame.
    Parses only the monthly section (6-digit YYYYMM rows).
    Values of -99.99 or -999 are replaced with NaN.
    Returns columns in col_names, converted % → decimal, indexed to _FULL_IDX.
    """
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    zf       = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    with zf.open(csv_name) as fh:
        raw = fh.read().decode("utf-8", errors="replace")

    data_lines = []
    for line in raw.splitlines():
        first = line.strip().split(",")[0].strip()
        if first.isdigit() and len(first) == 6:
            data_lines.append(line.strip())
        elif data_lines and not first.isdigit():
            break

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["date"] + col_names,
    )
    df["date"] = (
        pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m")
        + pd.offsets.MonthEnd(0)
    )
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    df = df.replace({-99.99: np.nan, -999.0: np.nan})
    df = df / 100
    return df.reindex(_get_full_idx())


def _align_to_er(arr: np.ndarray, T: int) -> np.ndarray:
    """Trim or pad a 1-D array from main_BC to length T (ER sample)."""
    if len(arr) >= T:
        return arr[-T:]
    return np.concatenate([np.full(T - len(arr), np.nan), arr])


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_common_growth_data() -> dict:
    """
    Load pi_t_mean, pi_ss, commonGrowth_mean, monthly_idx, nber_rec
    from commonGrowthData.npz (output of main_BC.py).

    Falls back to commonGrowthData_1965_2019.mat if npz not found.
    """
    npz_path = _DATA_DIR / "commonGrowthData.npz"
    mat_path = _DATA_DIR / "commonGrowthData_1965_2019.mat"

    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        print(f"  commonGrowthData: loaded from {npz_path.name}")
        return {
            "pi_t_mean":         data["pi_t_mean"],
            "pi_ss":             data["pi_ss"],
            "commonGrowth_mean": data["commonGrowth_mean"],
            "monthly_idx":       pd.DatetimeIndex(data["monthly_idx"])
                                 if "monthly_idx" in data else None,
            "nber_rec":          data["nber_rec"].astype(int)
                                 if "nber_rec" in data else None,
        }
    elif mat_path.exists():
        import scipy.io as sio
        mat = sio.loadmat(str(mat_path))
        print(f"  commonGrowthData: loaded from {mat_path.name} (legacy)")
        return {
            "pi_t_mean":         _squeeze(mat, "pi_t_mean"),
            "pi_ss":             _squeeze(mat, "pi_ss"),
            "commonGrowth_mean": _squeeze(mat, "commonGrowth_mean"),
            "monthly_idx":       None,
            "nber_rec":          None,
        }
    else:
        raise FileNotFoundError(
            f"\n  No commonGrowthData file found in {_DATA_DIR}"
            f"\n  Expected: {npz_path}"
            f"\n  Run main_BC.py first to generate commonGrowthData.npz"
        )


def load_return_data() -> dict:
    """
    Build return data from Ken French factor zips + commonGrowthData.npz.

    Returns
    -------
    dict with keys:
        marketReturn_excess  : (T,) Mkt-RF in decimal
        NBER_rec_index       : (T,) recession dummy 0/1 (from npz)
        marketRF             : (T,) RF in decimal
        marketReturn         : (T,) Mkt-RF + RF in decimal
        hmlFactor            : (T,) HML in decimal
        smbFactor            : (T,) SMB in decimal
        momentumFactor       : (T,) Mom in decimal
        conditionalVariance  : (T,) from npz cond_variance (NaN if not saved)
        date_data            : list of "YYYY-MM-DD" strings
        CAPE                 : None (not available without external data)
    """
    _FULL_IDX = _get_full_idx()
    T         = len(_FULL_IDX)
    print(f"  ER sample: {_FULL_IDX[0].strftime('%Y-%m')} → "
          f"{_FULL_IDX[-1].strftime('%Y-%m')}  ({T} months)")

    # ── Fama-French 3 factors: MktRF, SMB, HML, RF ───────────────────────────
    print("  Downloading FF 3 factors ...")
    ff     = _load_ff_zip(_FF_URL, ["MktRF", "SMB", "HML", "RF"])
    mkt_exc = ff["MktRF"]
    rf      = ff["RF"]
    mkt_ret = ff["MktRF"] + ff["RF"]
    smb     = ff["SMB"]
    hml     = ff["HML"]
    print(f"    Mkt-RF: {mkt_exc.notna().sum()} obs  "
          f"mean={mkt_exc.dropna().mean()*1200:.2f}% ann.")

    # ── Momentum factor ───────────────────────────────────────────────────────
    print("  Downloading FF Momentum factor ...")
    mom = _load_ff_zip(_MOM_URL, ["Mom"])["Mom"]
    print(f"    Mom:    {mom.notna().sum()} obs")

    # ── Load npz for NBER and conditional variance ────────────────────────────
    npz_path = _DATA_DIR / "commonGrowthData.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"\n  commonGrowthData.npz not found in {_DATA_DIR}"
            f"\n  Run main_BC.py first to generate it."
        )
    _npz = np.load(str(npz_path), allow_pickle=True)

    # NBER recession
    if "nber_rec" in _npz and "monthly_idx" in _npz:
        _nber_s = pd.Series(
            _npz["nber_rec"].astype(int),
            index=pd.DatetimeIndex(_npz["monthly_idx"]),
        ).reindex(_FULL_IDX).fillna(0).astype(int)
    else:
        raise KeyError(
            "nber_rec or monthly_idx not in commonGrowthData.npz — "
            "re-run main_BC.py."
        )
    print(f"    NBER:   {_nber_s.sum()} recession months")

    # ── Realized variance: Σ_d (r_d,t - r̄_t)²  in %² units ─────────────────
    # Following Moreira & Muir (2017) / Appendix A.10 of Gomez-Cram (2022).
    # Uses FF daily Mkt-RF (in %) — consistent with monthly equity returns source.
    # Returns kept in % before squaring → %² units matching author's .mat file
    # (spikes: ~20%² in Oct 1987, ~24%² in Oct 2008).
    print("  Computing realized variance from FF daily factors ...")
    try:
        # Download and parse daily FF data — do NOT reindex to monthly here
        _r_d = requests.get(_FF_DAILY_URL,
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        _r_d.raise_for_status()
        _zf_d    = zipfile.ZipFile(io.BytesIO(_r_d.content))
        _csv_d   = [n for n in _zf_d.namelist() if n.lower().endswith(".csv")][0]
        with _zf_d.open(_csv_d) as _fh:
            _raw_d = _fh.read().decode("utf-8", errors="replace")

        # Parse monthly section — daily rows have 8-digit YYYYMMDD dates
        _dl = []
        for _line in _raw_d.splitlines():
            _f = _line.strip().split(",")[0].strip()
            if _f.isdigit() and len(_f) == 8:
                _dl.append(_line.strip())
            elif _dl and not _f.isdigit():
                break

        _ff_d = pd.read_csv(
            io.StringIO("\n".join(_dl)), header=None,
            names=["date", "MktRF", "SMB", "HML", "RF"],
        )
        _ff_d["date"] = pd.to_datetime(
            _ff_d["date"].astype(str).str.strip(), format="%Y%m%d")
        _ff_d = _ff_d.set_index("date").apply(pd.to_numeric, errors="coerce")
        _ff_d = _ff_d.replace({-99.99: np.nan, -999.0: np.nan})
        # Keep in % (do NOT divide by 100) for %² variance units

        # Snap daily dates to month-end for groupby, then compute realized var
        _mkt_d = _ff_d["MktRF"].dropna().copy()
        _mkt_d.index = _mkt_d.index + pd.offsets.MonthEnd(0)
        _cvar_s = _mkt_d.groupby(level=0).apply(
            lambda x: ((x - x.mean()) ** 2).mean()   # mean not sum → matches author scale
        ).reindex(_FULL_IDX)
        _cvar = _cvar_s.values.astype(float)
        print(f"    cond_var: {np.sum(~np.isnan(_cvar))} obs  "
              f"mean={np.nanmean(_cvar):.4f}%²  "
              f"max={np.nanmax(_cvar):.4f}%² (expect ~1.2 in Oct-2008)")
    except Exception as e:
        _cvar = np.full(T, np.nan)
        print(f"    cond_var: ⚠ {e}")

    # ── CAPE ratio from Shiller's ie_data.xls ───────────────────────────────────
    # Date format in Shiller file: YYYY.MM (e.g. 1965.01)
    # CAPE is in column M (0-indexed: 12), header on row 8 (0-indexed: 7)
    print("  Downloading Shiller CAPE data ...")
    try:
        import io as _io
        _resp = requests.get(_SHILLER_URL,
                             headers={"User-Agent": "Mozilla/5.0"},
                             timeout=30)
        _resp.raise_for_status()
        _sh = pd.read_excel(
            _io.BytesIO(_resp.content),
            sheet_name="Data",
            header=7,           # row 8 (0-indexed) is the header
            usecols=[0, 12],    # Date and CAPE columns
        )
        _sh.columns = ["date", "CAPE"]
        # Parse Shiller date format YYYY.MM (e.g. 1965.01, 1965.1 = Oct)
        _sh["date"] = _sh["date"].astype(str).str.strip()
        _sh = _sh[_sh["date"].str.match(r"^\d{4}\.\d+$", na=False)].copy()
        def _parse_shiller_date(d):
            yr, mo_str = d.split(".")
            # Handle single digit month: "1" → Jan, "10" → Oct
            mo = int(mo_str) if len(mo_str) >= 2 else int(mo_str) * 10
            return pd.Timestamp(int(yr), min(mo, 12), 1) + pd.offsets.MonthEnd(0)
        _sh["date"] = _sh["date"].apply(_parse_shiller_date)
        _sh = _sh.set_index("date")["CAPE"].apply(
            pd.to_numeric, errors="coerce").sort_index()
        _cape = _sh.reindex(_FULL_IDX).values.astype(float)
        print(f"    CAPE: {np.sum(~np.isnan(_cape))} obs  "
              f"mean={np.nanmean(_cape):.2f}  "
              f"range=[{np.nanmin(_cape):.1f}, {np.nanmax(_cape):.1f}]")
    except Exception as e:
        _cape = np.full(T, np.nan)
        print(f"    CAPE: ⚠ {e}")

    # ── Save all collected data to CSV ───────────────────────────────────────
    _out = pd.DataFrame({
        "marketReturn_excess": mkt_exc.values,
        "marketRF":            rf.values,
        "marketReturn":        mkt_ret.values,
        "NBER_rec_index":      _nber_s.values,
        "hmlFactor":           hml.values,
        "smbFactor":           smb.values,
        "momentumFactor":      mom.values,
        "conditionalVariance": _cvar,
        "CAPE":                _cape,
    }, index=_FULL_IDX)
    _out.index.name = "date"
    _csv_path = _DATA_DIR / "returnData.csv"
    _out.to_csv(_csv_path)
    print(f"  Saved → {_csv_path}")

    return {
        "marketReturn_excess": mkt_exc.values.astype(float),
        "NBER_rec_index":      _nber_s.values.astype(float),
        "marketRF":            rf.values.astype(float),
        "marketReturn":        mkt_ret.values.astype(float),
        "hmlFactor":           hml.values.astype(float),
        "smbFactor":           smb.values.astype(float),
        "momentumFactor":      mom.values.astype(float),
        "conditionalVariance": _cvar.astype(float),
        "date_data":           [d.strftime("%Y-%m-%d") for d in _FULL_IDX],
        "CAPE":                _cape.astype(float),
    }


def load_initial_parameters() -> np.ndarray:
    """
    Load para.txt — 7-element starting parameter vector.
    [mu_0, rho, corr_s, phi_1, phi_2, h, sigma2_1]
    """
    p = _DATA_DIR / "para.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"para.txt not found in {_DATA_DIR}\n"
            f"This file contains the 7 starting parameter values."
        )
    para = np.atleast_1d(np.loadtxt(str(p))).reshape(-1).astype(float)
    if len(para) != 7:
        raise ValueError(
            f"para.txt must contain exactly 7 values; found {len(para)}."
        )
    return para


# ---------------------------------------------------------------------------
# Main convenience loader — called by main_ERF.py
# ---------------------------------------------------------------------------

def load_expected_return_data(base_dir=None) -> tuple:
    """
    Load all estimation inputs and return as a tuple.

    `base_dir` is accepted but ignored — paths are resolved relative to
    this file's location so the loader works on any machine.

    pi_t and z_t from main_BC (e.g. 840 obs in replicate mode) are aligned
    to the ER sample (T obs) by taking the last T observations, since both
    samples share the same end date.

    sigma2_1_fix = std(YY[expansion])² with ddof=1, matching MATLAB.

    Returns
    -------
    YY           : (T, 1) ndarray  — excess returns r^e_{1:T}
    rf           : (T,)   ndarray  — monthly risk-free rate (decimal)
    NBERIndex    : (T,)   ndarray  — NBER recession dummy (0/1)
    pi_t         : (T,)   ndarray  — filtered recession probs π̂_{t|t}
    z_t          : (T,)   ndarray  — common growth factor from main_BC
    sigma2_1_fix : float           — variance of expansion-period returns
    """
    ret       = load_return_data()
    cgd       = load_common_growth_data()
    T         = len(_get_full_idx())

    YY        = ret["marketReturn_excess"].reshape(-1, 1)
    rf        = ret["marketRF"]
    NBERIndex = ret["NBER_rec_index"]
    pi_t      = _align_to_er(cgd["pi_t_mean"],         T)
    z_t       = _align_to_er(cgd["commonGrowth_mean"],  T)

    print(f"  pi_t: {len(pi_t)} obs  "
          f"range=[{pi_t.min():.4f}, {pi_t.max():.4f}]")
    print(f"  z_t:  {len(z_t)} obs  "
          f"range=[{z_t.min():.4f}, {z_t.max():.4f}]")

    expansion_mask = (NBERIndex == 0)
    sigma2_1_fix   = float(np.std(YY[expansion_mask, 0], ddof=1) ** 2)

    print(f"  T={T}  recession months={int(NBERIndex.sum())}"
          f"  sigma2_1_fix={sigma2_1_fix:.8f}")

    return YY, rf, NBERIndex, pi_t, z_t, sigma2_1_fix