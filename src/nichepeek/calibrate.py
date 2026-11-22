"""Calibration of the first-level estimate with semi-synthetic data built from real tissue.

A calibration table has one row per semi-synthetic setting: tissue, variant, labeler, p0 (first thinning level),
ladder_depth (counts per cell the ladder sees), ratio_T (median first-level estimate / median measured own-label
bias) and optionally ratio_T_lo / ratio_T_hi (bootstrap over independent units). The package looks for
nichepeek/data/calibration.csv; build one from your own semi-synthetic runs
(make_calibration_table.py in the analysis scripts deposited with the paper), or pass any table with --table.

Matching: rows with the same first step (|p0 - p0_run| <= 0.011) and the centroid labeler (of the given tissue,
if requested and present); the divisor is the row whose ladder depth is closest on a log scale. The interval is that
row's bootstrap interval over independent units; the spread over all rows with the same first step is reported
separately. A run outside the depth range of the matching rows is flagged.

Truth correction. The truth labels of semi-synthetic data have errors of their own, which shrink the measured
bias and so make ratio_T too large. Optional columns give a divisor corrected for this by truth-side extrapolation
(the paper's semi-synthetic scripts): ratio_T_corr, ratio_T_corr_lo / _hi (bootstrap over units), corr_usable (False where
thinning the truth moved its flip rate too little for a reliable extrapolation) and corr_note. When the matching row
has a usable corrected divisor, calibrated_T_corr (and _lo/_hi, calibrated_share_corr) are added beside the
uncorrected value; the two bracket the bias. Tables without these columns work as before.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DEFAULT = os.path.join(os.path.dirname(__file__), "data", "calibration.csv")


def load_table(path=None):
    path = path or DEFAULT
    if not os.path.exists(path):
        raise FileNotFoundError(f"no calibration table at {path}: build one with make_calibration_table.py "
                                "or pass --table")
    return _prep(pd.read_csv(path))


def _prep(T):
    T = T.copy()
    need = {"tissue", "variant", "p0", "ladder_depth", "ratio_T"}
    if not need <= set(T.columns):
        raise ValueError(f"calibration table needs columns {sorted(need)}")
    if "labeller" not in T:
        T["labeller"] = np.where(T.variant.astype(str).str.contains("logreg"), "logreg", "centroid")
    return T


def calibrate(o, table=None, tissue=None):
    """o: the 'overall' dict of a run (needs first_level_T, first_level_p, ladder_depth, program_published).
    tissue: prefer calibration rows of this tissue (e.g. 'xenium'), if the table has any.
    Returns calibrated_* keys: the point value (divisor of the matching row), its interval from that row's
    bootstrap over independent units (calibrated_T_lo/hi), and the spread over all rows with the same first step
    (calibrated_T_rows_lo/hi). NaN with a reason when no row matches."""
    T = _prep(table) if isinstance(table, pd.DataFrame) else load_table(table)
    out = dict(calibrated_T=np.nan, calibrated_T_lo=np.nan, calibrated_T_hi=np.nan, calibrated_share=np.nan)
    e, p0 = o.get("first_level_T", np.nan), o.get("first_level_p", np.nan)
    if not np.isfinite(e):
        out["calibration_note"] = "no first-level estimate in this run"
        return out
    m = T[(np.abs(T.p0 - p0) <= 0.011) & (T.labeller == "centroid")]
    if m.empty:
        out["calibration_note"] = (f"no calibration row with first step p = {p0:g}; rerun with a first step "
                                   f"available in the table ({', '.join(f'{v:g}' for v in sorted(T.p0.unique()))})")
        return out
    cand = m[m.tissue == tissue] if tissue is not None and (m.tissue == tissue).any() else m
    d = np.abs(np.log(cand.ladder_depth) - np.log(o["ladder_depth"]))
    r = cand.iloc[int(np.argmin(d.to_numpy()))]
    lo_r = r.get("ratio_T_lo", r.ratio_T); hi_r = r.get("ratio_T_hi", r.ratio_T)
    out.update(calibrated_T=float(e / r.ratio_T), calibrated_T_lo=float(e / hi_r), calibrated_T_hi=float(e / lo_r),
               calibrated_T_rows_lo=float(e / m.ratio_T.max()), calibrated_T_rows_hi=float(e / m.ratio_T.min()),
               calibration_divisor=float(r.ratio_T), calibration_source=f"{r.tissue} {r.variant}",
               calibration_depth=float(r.ladder_depth),
               calibration_extrapolated=bool(o["ladder_depth"] < m.ladder_depth.min() / 1.25 or
                                             o["ladder_depth"] > m.ladder_depth.max() * 1.25))
    prog = o.get("program_published", np.nan)
    out["calibrated_share"] = float(out["calibrated_T"] / prog) if prog else np.nan
    out.update(_corrected(e, r, prog))
    return out


def _num(r, k):
    v = r.get(k, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _corrected(e, r, prog):
    """Truth-corrected calibration from the matching row, if the table has it."""
    rc = _num(r, "ratio_T_corr")
    if not np.isfinite(rc):
        return dict(calibration_corr_note="no truth-corrected divisor for this calibration row")
    usable = r.get("corr_usable", True)
    usable = str(usable).strip().lower() not in ("false", "0", "no", "nan", "")
    if not usable:
        note = r.get("corr_note", "")
        return dict(calibration_corr_usable=False,
                    calibration_corr_note=f"truth correction not usable for {r.tissue} {r.variant}"
                                          + (f": {note}" if isinstance(note, str) and note else ""))
    lo_r, hi_r = _num(r, "ratio_T_corr_lo"), _num(r, "ratio_T_corr_hi")
    lo_r = lo_r if np.isfinite(lo_r) else rc
    hi_r = hi_r if np.isfinite(hi_r) else rc
    out = dict(calibration_corr_usable=True, calibration_divisor_corr=rc, calibrated_T_corr=float(e / rc),
               calibrated_T_corr_lo=float(e / hi_r), calibrated_T_corr_hi=float(e / lo_r))
    out["calibrated_share_corr"] = float(out["calibrated_T_corr"] / prog) if prog else np.nan
    return out


def calibrate_dir(result_dir, table=None, tissue=None):
    """Calibrate a finished run: reads result.json, writes calibration.json next to it, returns the dict."""
    o = json.load(open(os.path.join(result_dir, "result.json")))["overall"]
    c = calibrate(o, table, tissue)
    json.dump(c, open(os.path.join(result_dir, "calibration.json"), "w"), indent=1, default=float)
    return c
