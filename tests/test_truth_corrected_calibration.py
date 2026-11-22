"""Truth-corrected calibration (columns ratio_T_corr, corr_usable) beside the uncorrected one."""
import numpy as np
import pandas as pd
import pytest

from nichepeek.calibrate import calibrate, load_table

O = dict(first_level_T=0.0045, first_level_p=0.8, ladder_depth=147, program_published=0.065)


def _table(**extra):
    T = pd.DataFrame(dict(tissue=["xenium", "liver"], variant=["deep", "deep"], p0=[0.8, 0.8],
                          ladder_depth=[74, 90], ratio_T=[0.69, 0.70]))
    return T.assign(**extra)


def test_without_correction_columns_unchanged():
    c = calibrate(O, _table(), tissue="xenium")
    assert c["calibrated_T"] == pytest.approx(0.0045 / 0.69)
    assert "calibrated_T_corr" not in c and "no truth-corrected divisor" in c["calibration_corr_note"]


def test_usable_correction_reported_beside_uncorrected():
    T = _table(ratio_T_corr=[0.29, 0.13], ratio_T_corr_lo=[0.2, np.nan], ratio_T_corr_hi=[0.35, np.nan],
               corr_usable=[True, False], corr_note=["", "truth flip rose too little"])
    c = calibrate(O, T, tissue="xenium")
    assert c["calibrated_T"] == pytest.approx(0.0045 / 0.69)                      # uncorrected unchanged
    assert c["calibrated_T_corr"] == pytest.approx(0.0045 / 0.29)
    assert c["calibrated_T_corr_lo"] == pytest.approx(0.0045 / 0.35)
    assert c["calibrated_T_corr_hi"] == pytest.approx(0.0045 / 0.2)
    assert c["calibrated_share_corr"] > c["calibrated_share"] and c["calibration_corr_usable"]
    c2 = calibrate(O, T, tissue="liver")
    assert "calibrated_T_corr" not in c2 and not c2["calibration_corr_usable"]
    assert "not usable" in c2["calibration_corr_note"] and "too little" in c2["calibration_corr_note"]


def test_usable_flag_read_from_csv(tmp_path):
    f = tmp_path / "t.csv"
    _table(ratio_T_corr=[0.29, 0.13], corr_usable=[True, False]).to_csv(f, index=False)
    T = load_table(str(f))
    assert np.isfinite(calibrate(O, T, tissue="xenium")["calibrated_T_corr"])
    assert "calibrated_T_corr" not in calibrate(O, T, tissue="liver")


def test_packaged_table_loads():
    T = load_table()
    assert {"tissue", "variant", "p0", "ratio_T"} <= set(T.columns)


def test_report_mentions_both_values(tmp_path):
    import warnings
    warnings.filterwarnings("ignore")
    import nichepeek as na
    from nichepeek.simulate import make_anndata
    A = make_anndata(n_sections=3, seed=4, n_cells=2000, depth=70, label_noise=0.1)
    f = tmp_path / "cal.csv"
    pd.DataFrame(dict(tissue=["sim"], variant=["deep"], p0=[0.8], ladder_depth=[35], ratio_T=[0.7],
                      ratio_T_corr=[0.3], corr_usable=[True])).to_csv(f, index=False)
    r = na.diagnose(A, "label", "^F", "section", out_dir=str(tmp_path / "run"), spatial=("x", "y"),
                    counts_layer="counts", verbose=False, p=(0.8, 0.5), calibrate=str(f))
    txt = " ".join(r.interpretation)
    assert "uncorrected" in txt and "bracket" in txt
