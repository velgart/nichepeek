"""Tests for n_perm, min_depth, the flips command and the interpretation wording."""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
import nichepeek as na
from nichepeek.config import Config
from nichepeek.ladder import ladder
from nichepeek.simulate import TGT, make_anndata, section


def test_default_cache_key_unchanged_by_new_fields():
    # the key of a default Config must not depend on n_perm (when 1) or min_depth, so cached runs stay valid
    assert Config().key() == Config(n_perm=1, min_depth=5.0).key()
    assert Config().key() != Config(n_perm=20).key()


def test_n_perm_1_reproduces_single_permutation():
    rng = np.random.default_rng(0)
    xy, lab, Y = section(rng, n_cells=3000, depth=70)
    t = np.isin(lab, TGT)
    a = ladder(xy, lab, Y, 5, t, noise="multinomial")
    b = ladder(xy, lab, Y, 5, t, noise="multinomial", n_perm=1)
    assert a == b or all(np.isclose(a[k], b[k], equal_nan=True) for k in a)
    c = ladder(xy, lab, Y, 5, t, noise="multinomial", n_perm=10)
    assert c["S0"] == a["S0"] and c["S3"] == a["S3"] and c["S1"] != a["S1"]


def test_n_perm_reduces_s1_noise():
    rng = np.random.default_rng(3)
    xy, lab, Y = section(rng, n_cells=4000, depth=70)
    t = np.isin(lab, TGT)
    one = [ladder(xy, lab, Y, s, t, noise="multinomial")["S1"] for s in range(8)]
    many = [ladder(xy, lab, Y, s, t, noise="multinomial", n_perm=10)["S1"] for s in range(8)]
    assert np.std(many) < np.std(one)


@pytest.fixture(scope="module")
def sim():
    return make_anndata(n_sections=3, seed=0, n_cells=2500, depth=70)


def test_min_depth_levels_and_warning(sim):
    r = na.diagnose(sim, "label", "^F", "section", spatial=("x", "y"), counts_layer="counts", verbose=False,
                    p=(0.5, 0.25, 0.12))
    o = r.overall
    assert o["n_levels_used"] == sum(r.summary.effective_depth >= 10)
    assert "lower_bound_T_all_levels" in o
    if o["levels_below_min_depth"]:
        assert any(l.startswith("WARNING: thinning level") for l in r.interpretation)
    assert any(l.startswith("Destination:") for l in r.interpretation)


def test_flips(sim):
    from nichepeek.flips import flip_rates
    D = flip_rates(sim, "label", "^F", "section", spatial=("x", "y"), counts_layer="counts", crop_cells=1500,
                   verbose=False)
    assert {"flip_full_pub", "flip_half_rc", "crop_overlap"} <= set(D.columns)
    assert (D.crop_overlap > 0).all()                      # 2 crops of 1500 from ~2500 cells must overlap
    assert D.flip_full_pub.median() <= D.flip_half_rc.median() + 0.02   # full depth is at least as stable


def test_logreg_labeller_runs_and_is_reproducible(sim):
    import nichepeek.io as io
    from nichepeek.labeling import LogRegLabeler
    import pandas as pd
    data = io.load(sim, "label", "section", spatial=("x", "y"), counts_layer="counts")
    vc = pd.Series(data.label).value_counts()
    labels = np.array(sorted(vc[vc >= 50].index))
    keep = np.isin(data.label, labels)
    a = LogRegLabeler.from_data(data, labels, keep, seed=0)
    b = LogRegLabeler.from_data(data, labels, keep, seed=0)
    ii = np.where(data.parent == data.parent[0])[0][:500]
    Y = data.counts[ii].toarray()
    pa, pb = a.predict(Y, data.parent[0]), b.predict(Y, data.parent[0])
    assert (pa == pb).all()                                   # same seed, same labels (no hash randomisation)
    assert set(pa) <= set(labels)
    assert (pa == data.label[ii]).mean() > 0.5


def test_logreg_labeller_has_own_cache_key(sim, tmp_path):
    for lab in ("centroid", "logreg"):
        na.diagnose(sim, "label", "^F", "section", spatial=("x", "y"), counts_layer="counts", labeler=lab,
                    out_dir=str(tmp_path / lab), p=(0.5,), ncem=False, context_tests=False, verbose=False)
    import os
    ka = os.listdir(tmp_path / "centroid" / "sections")[0]
    kb = os.listdir(tmp_path / "logreg" / "sections")[0]
    assert ka != kb


def test_drift(sim):
    from nichepeek.drift import label_drift
    D = label_drift(sim, "label", "^F", "section", spatial=("x", "y"), counts_layer="counts", p=(0.5, 0.25),
                    verbose=False)
    assert {"entropy", "largest_share", "dissim_added", "dissim_native", "confusion_corr"} <= set(D.columns)
    assert (D.groupby("section").size() == 2).all()


def test_drift_logreg_labeller_is_used(sim):
    # `drift --labeler logreg` used to fall back silently to the centroid labeler
    from nichepeek.drift import label_drift
    kw = dict(spatial=("x", "y"), counts_layer="counts", p=(0.5,), verbose=False)
    Dc = label_drift(sim, "label", "^F", "section", **kw)
    Dl = label_drift(sim, "label", "^F", "section", labeler="logreg", **kw)
    assert not Dc[["added_flip", "entropy"]].equals(Dl[["added_flip", "entropy"]])
    with pytest.raises(ValueError):
        label_drift(sim, "label", "^F", "section", labeler="nope", **kw)


def test_cli_drift_passes_labeller(sim, tmp_path, monkeypatch):
    import nichepeek.drift as dr
    from nichepeek import cli
    seen = {}
    def fake(*a, **k):
        import pandas as pd
        seen.update(k)
        return pd.DataFrame(dict(p=[0.5], added_flip=[0.1], largest_share=[0.5], entropy=[1.0], dissim_added=[0.1],
                                 confusion_corr=[0.5], entropy_c1=[1.0], dissim_native=[0.1]))
    monkeypatch.setattr(dr, "label_drift", fake)
    cli.main(["drift", "x.h5ad", "--label-key", "label", "--target", "^F", "--section-key", "section",
              "--labeler", "logreg", "--out", str(tmp_path / "d.csv")])
    assert seen["labeler"] == "logreg"


def test_labels_file_matches_obs_and_leaves_data_untouched(sim, tmp_path):
    import hashlib
    import pandas as pd
    f = tmp_path / "d.h5ad"
    sim.write_h5ad(f)
    before = hashlib.sha256(f.read_bytes()).hexdigest()
    pd.DataFrame({"alt": sim.obs["label"].astype(str)}, index=sim.obs_names).to_csv(tmp_path / "lab.csv")
    kw = dict(spatial=("x", "y"), counts_layer="counts", p=(0.5,), ncem=False, context_tests=False, verbose=False)
    a = na.diagnose(str(f), "label", "^F", "section", out_dir=str(tmp_path / "a"), **kw)
    b = na.diagnose(str(f), "alt", "^F", "section", labels_file=str(tmp_path / "lab.csv"), out_dir=str(tmp_path / "b"), **kw)
    x, y = a.sections.select_dtypes("number"), b.sections.select_dtypes("number")
    assert np.allclose(x.to_numpy(float), y[x.columns].to_numpy(float), equal_nan=True)
    assert hashlib.sha256(f.read_bytes()).hexdigest() == before
