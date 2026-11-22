"""First-level estimate in the report, calibration, clipped-gene share, run time/memory."""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
import nichepeek as na
from nichepeek.calibrate import calibrate
from nichepeek.simulate import make_anndata

KW = dict(spatial=("x", "y"), counts_layer="counts", verbose=False, p=(0.8, 0.5))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    A = make_anndata(n_sections=3, seed=4, n_cells=2500, depth=70, label_noise=0.1)
    d = str(tmp_path_factory.mktemp("r024"))
    return na.diagnose(A, "label", "^F", "section", out_dir=d, **KW), d


def test_first_level_clip_memory_and_wording(run):
    r, _ = run
    o = r.overall
    assert o["first_level_p"] == 0.8 and np.isfinite(o["first_level_T"])
    assert 0 <= o["clip_frac_S0"] <= 1 and 0 <= o["clip_frac_S1"] <= 1
    assert o["peak_memory_gb"] > 0 and o["seconds"] >= 0
    txt = " ".join(r.interpretation).lower()
    assert "lower bound" not in txt and "first-level estimate" in txt and "clipping" in txt


def test_calibrate_matching():
    T = pd.DataFrame(dict(tissue=["a", "a", "b"], variant=["deep", "shallow", "x"], p0=[0.8, 0.8, 0.5],
                          ladder_depth=[80, 35, 80], ratio_T=[0.7, 0.45, 1.3]))
    o = dict(first_level_T=0.007, first_level_p=0.8, ladder_depth=70, program_published=0.07)
    c = calibrate(o, T)
    assert c["calibration_divisor"] == 0.7 and abs(c["calibrated_T"] - 0.01) < 1e-12
    assert abs(c["calibrated_share"] - 0.01 / 0.07) < 1e-12
    assert c["calibrated_T_lo"] == pytest.approx(0.01) and c["calibrated_T_hi"] == pytest.approx(0.01)   # no bootstrap cols
    assert c["calibrated_T_rows_lo"] == pytest.approx(0.007 / 0.7) and c["calibrated_T_rows_hi"] == pytest.approx(0.007 / 0.45)
    T2 = T.assign(ratio_T_lo=[0.5, 0.3, 1.0], ratio_T_hi=[0.9, 0.6, 1.5], tissue=["a", "b", "b"])
    c3 = calibrate(dict(o, ladder_depth=40), T2, tissue="a")          # tissue preferred over nearer depth
    assert c3["calibration_source"] == "a deep" and c3["calibrated_T_lo"] == pytest.approx(0.007 / 0.9)
    assert not c["calibration_extrapolated"]
    c2 = calibrate(dict(o, first_level_p=0.6), T)
    assert np.isnan(c2["calibrated_T"]) and "first step" in c2["calibration_note"]
    assert calibrate(dict(o, ladder_depth=300), T)["calibration_extrapolated"]


def test_report_rederives_old_results(run):
    from nichepeek.cli import main
    _, d = run
    f = os.path.join(d, "result.json"); j = json.load(open(f))
    for k in ["first_level_T", "first_level_p", "clip_frac_S0"]:
        j["overall"].pop(k, None)
    j["interpretation"] = ["Lower bound on native label-error bias ..."]
    json.dump(j, open(f, "w"))
    assert main(["report", d]) == 0
    j2 = json.load(open(f))
    assert "first_level_T" in j2["overall"] and not any("Lower bound" in l for l in j2["interpretation"])
