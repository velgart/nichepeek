import numpy as np
import pandas as pd
import pytest

import nichepeek as na


def test_outputs(result):
    R, S = result.sections, result.summary
    assert len(R) == 3 and list(S.p) == [0.6, 0.35]
    for col in ["L_program_0.35", "L_T_program_0.35", "L_N_program_0.35", "L_M_program_0.35",
                "circ_identity_native", "x_0.35", "flip_native"]:
        assert col in R, col
    assert {"toward_context", "toward_null", "auc_gain"} <= set(S.columns)
    assert result.interpretation


def test_exact_identities(result):
    R = result.sections
    for p in (0.6, 0.35):
        assert np.all(R[f"L_T_S0_{p}"] == 0)                                   # own-label channel: S0 untouched
        lhs = R[f"L_program_{p}"]
        rhs = -R[f"L_identity_{p}"] + R[f"L_S0_{p}"] - R[f"L_S3_{p}"]
        assert np.allclose(lhs, rhs, atol=1e-12)
        tn = R[f"L_program_{p}"] - R[f"L_T_program_{p}"] - R[f"L_N_program_{p}"]
        assert np.allclose(tn, R[f"L_TN_program_{p}"], atol=1e-12)


def test_known_truth_directions(result):
    R = result.sections
    assert R.circ_identity_native.median() > 0          # labels fitted to their own counts gain identity
    assert R["L_T_program_0.35"].median() > 0           # own-label error inflates program (simulation truth)


def test_naive_design_runs(adata):
    r = na.diagnose(adata, "label", "^F", "section", counts_layer="counts", p=(0.5,), split=False,
                    fixed_targets=False, channels=False, membership=False, context_tests=False, verbose=False)
    assert "circ_identity_native" not in r.sections and "L_program_0.5" in r.sections


def test_custom_labeler(adata):
    from nichepeek.labeling import classify, norm_log
    genes_ref = np.asarray(adata.layers["counts"].toarray())
    lab = adata.obs.label.to_numpy()
    labels = sorted(set(lab))
    cent = np.vstack([norm_log(genes_ref[lab == l], 60).mean(0) for l in labels])
    f = lambda Y, section: classify(norm_log(Y, 60), cent, labels)       # a user-supplied labeler
    r = na.diagnose(adata, "label", "^F", "section", counts_layer="counts", p=(0.5,), labeler=f,
                    context_tests=False, verbose=False)
    assert np.isfinite(r.sections["L_program_0.5"]).all()


def test_file_input_and_parallel_match_memory(adata, result, tmp_path):
    f = str(tmp_path / "toy.h5ad")
    adata.write_h5ad(f)
    r = na.diagnose(f, "label", "^F", "section", counts_layer="counts", p=(0.6, 0.35), n_jobs=2, verbose=False)
    pd.testing.assert_frame_equal(r.sections, result.sections)


def test_resume(adata, tmp_path, monkeypatch):
    kw = dict(counts_layer="counts", p=(0.5,), context_tests=False, verbose=False, out_dir=str(tmp_path / "o"))
    r1 = na.diagnose(adata, "label", "^F", "section", **kw)
    import nichepeek.api as api
    monkeypatch.setattr(api, "_run_one", lambda args: (_ for _ in ()).throw(RuntimeError("recomputed")))
    r2 = na.diagnose(adata, "label", "^F", "section", **kw)            # everything from the cache
    pd.testing.assert_frame_equal(r1.sections, r2.sections)
    assert (tmp_path / "o" / "report.pdf").exists()


def test_ncem_and_noise_models(adata):
    kw = dict(counts_layer="counts", p=(0.6,), context_tests=False, verbose=False)
    r = na.diagnose(adata, "label", "^F", "section", ncem=True, ncem_genes=60, **kw)
    R = r.sections
    for arm in ["a", "c1", "b0.6", "c0.6", "T0.6", "N0.6"]:               # non-circular arms: a proper R2
        assert (R[f"{arm}_ncem_full"] <= 1).all(), arm
    # own-label channel leaves the neighbors-only model untouched: type and full move identically
    assert np.allclose(R["L_T_ncem_type_0.6"], R["L_T_ncem_full_0.6"], atol=1e-12)
    # neighbor channel leaves the own-label-only model untouched: niche and full move identically
    assert np.allclose(R["L_N_ncem_niche_0.6"], R["L_N_ncem_full_0.6"], atol=1e-12)
    rp = na.diagnose(adata, "label", "^F", "section", noise="poisson", **kw)
    assert r.config.noise == "multinomial" and rp.config.noise == "poisson"
    assert not np.allclose(R["a_identity"], rp.sections["a_identity"])


def test_cache_invalidated_by_input_settings(adata, tmp_path):
    kw = dict(counts_layer="counts", p=(0.6,), context_tests=False, verbose=False, out_dir=str(tmp_path / "o"))
    na.diagnose(adata, "label", "^F", "section", **kw)
    r = na.diagnose(adata, "label", "^F", "section", exclude_genes=r"^(?:g0|g1)$", **kw)   # other genes
    assert r.meta["n_genes"] == 148
    assert len(list((tmp_path / "o" / "sections").iterdir())) == 2                       # separate caches


def test_anchored_alias_pattern():
    import re
    from nichepeek.io import DEFAULT_EXCLUDE
    rx = re.compile(DEFAULT_EXCLUDE + "|^(?:Dclk1|Gdf1)$")
    hits = [g for g in ["Blank-12", "Dclk1", "Gdf1", "Gdf10", "Gdf11", "Gdf15", "Dclk1b", "Col1a1"] if rx.match(g)]
    assert hits == ["Blank-12", "Dclk1", "Gdf1"]


def test_n_splits_average(adata):
    kw = dict(counts_layer="counts", p=(0.6,), context_tests=False, verbose=False)
    r1 = na.diagnose(adata, "label", "^F", "section", **kw)
    r3 = na.diagnose(adata, "label", "^F", "section", n_splits=3, **kw)
    R1, R3 = r1.sections, r3.sections
    assert (R3.n_splits == 3).all() and (R3["sd_L_program_0.6"] > 0).all()
    assert R3.n_cells.dtype.kind == "i" and (R3.n_cells == R1.n_cells).all()   # counts stay integers
    assert not np.allclose(R1["L_program_0.6"], R3["L_program_0.6"])       # averaged over other splits
    assert np.allclose(R1["a_identity"], R1["a_identity"])
    # the exact identities hold for the average too (they are linear in the arms)
    assert np.all(R3["L_T_S0_0.6"] == 0)
    assert np.allclose(R3["L_program_0.6"] - R3["L_T_program_0.6"] - R3["L_N_program_0.6"], R3["L_TN_program_0.6"])
