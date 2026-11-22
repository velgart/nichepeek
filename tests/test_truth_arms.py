"""Validation arms with known true labels (--truth-key)."""
import glob
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
import nichepeek as na
from nichepeek.simulate import make_anndata

KW = dict(spatial=("x", "y"), counts_layer="counts", verbose=False, p=(0.5, 0.25))


@pytest.fixture(scope="module")
def sim():
    A = make_anndata(n_sections=3, seed=2, n_cells=2500, depth=70, label_noise=0.1)
    A.obs["pub_copy"] = A.obs["label"].astype(str)
    return A


@pytest.fixture(scope="module")
def runs(sim):
    base = na.diagnose(sim, "label", "^F", "section", **KW)
    same = na.diagnose(sim, "label", "^F", "section", truth_key="pub_copy", **KW)
    true = na.diagnose(sim, "label", "^F", "section", truth_key="true_label", **KW)
    return base, same, true


def test_truth_arms_leave_other_results_unchanged(runs):
    base, same, _ = runs
    cols = [c for c in base.sections.columns if c.startswith(("L_", "circ_", "a_", "c1_", "x_", "flip"))]
    pd.testing.assert_frame_equal(base.sections[cols], same.sections[cols])
    assert not any(c.startswith("true_") for c in base.sections.columns)


def test_truth_equal_to_published_reduces_to_existing_arms(runs):
    _, same, _ = runs
    R = same.sections
    for q in ["program", "identity"]:
        np.testing.assert_allclose(R[f"true_net_{q}"], R[f"c1_{q}"] - R[f"a_{q}"], atol=1e-12)
        np.testing.assert_allclose(R[f"true_T_{q}"], R[f"T1_{q}"] - R[f"a_{q}"], atol=1e-12)
        np.testing.assert_allclose(R[f"true_pub_net_{q}"], 0.0, atol=1e-12)
    assert (R.acc_published_vs_truth_targets == 1).all()


def test_measured_own_label_bias_positive_with_real_truth(runs):
    # simulations: the own-label native bias is positive in every regime
    _, _, true = runs
    assert true.overall["true_T_program"] > 0
    assert true.overall["acc_c1_vs_truth_targets"] < 1


def test_cache_key_without_truth_unchanged(sim, tmp_path):
    # the input key of runs without truth_key does not depend on the truth option
    na.diagnose(sim, "label", "^F", "section", out_dir=str(tmp_path), **KW)
    d = os.path.basename(glob.glob(str(tmp_path / "sections" / "*"))[0])
    ikey_old = hashlib.sha1(json.dumps(dict(
        src="memory", label_key="label", target="^F", labels_file=None, section_key="section", spatial=("x", "y"),
        counts_layer="counts", crop_cells=0, crops_per_section=2, exclude_genes=na.io.DEFAULT_EXCLUDE, n_genes=150,
        labeler="CentroidLabeler"), sort_keys=True, default=str).encode()).hexdigest()[:8]
    assert d.endswith("_" + ikey_old)
