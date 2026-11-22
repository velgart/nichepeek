"""Report reading gated by target identity and sign consistency, destination test with profiles of all
labels, sections with too few targets skipped; per-section results and cache keys unchanged."""
import numpy as np
import pandas as pd
import pytest


def test_consistent_sign():
    from nichepeek.summarize import consistent_sign
    assert consistent_sign(12, 12) == 1 and consistent_sign(9, 12) == 1 and consistent_sign(8, 12) == 0
    assert consistent_sign(0, 10) == -1 and consistent_sign(2, 10) == -1 and consistent_sign(4, 7) == 0
    assert consistent_sign(3, 3) == 0                       # fewer than four units: no direction called


def test_min_targets_not_in_cache_key():
    from nichepeek.config import Config
    assert Config(min_targets=5).key() == Config().key() == Config(min_targets=500).key()


def _S(t, k):
    return pd.DataFrame({"p": [0.8, 0.6, 0.4], "effective_depth": [40.0, 30.0, 20.0], "L_T_program": [t] * 3,
                         "L_N_program": [-0.001] * 3, "L_program": [t] * 3, "n_T_pos": [k] * 3})


def _o(I=0.1, n_tl=3, n=8, k=8):
    return dict(n_sections=n, identity_published=I, n_target_labels=n_tl, first_level_T=0.004, first_level_p=0.8,
                first_level_T_n_pos=k, program_published=0.05)


def test_interpret_wording():
    from nichepeek.summarize import interpret
    txt = " ".join(interpret(_o(), _S(0.004, 8)))
    assert "increased the estimated neighborhood share" in txt and "Its sign was reliable" in txt
    txt = " ".join(interpret(_o(k=4), _S(0.0001, 4)))       # median > 0 but 4 of 8 positive
    assert "no consistent direction" in txt and "increased" not in txt and "undetermined" in txt
    txt = " ".join(interpret(_o(n_tl=1), _S(0.0001, 5)))
    assert "single target label" in txt and "not defined for a single target label" in txt
    txt = " ".join(interpret(_o(I=0.005), _S(0.001, 7)))
    assert "identity is low" in txt and "Its sign was reliable" not in txt


def test_single_label_destination_and_skip(adata):
    import nichepeek as na
    lab = adata.obs["label"].astype(str).to_numpy()
    tgt = sorted(l for l in set(lab) if l.startswith("F"))[0]
    r = na.diagnose(adata, "label", f"^{tgt}$", "section", counts_layer="counts", p=(0.6, 0.35), verbose=False)
    assert r.overall["n_target_labels"] == 1
    assert any("single target label" in s for s in r.interpretation)
    assert (r.summary["n_flips_all"] > 0).all() and np.isfinite(r.summary["toward_context_all"]).all()
    # a section with too few targets is skipped and named
    A = adata.copy()
    sec = A.obs["section"].astype(str).to_numpy()
    s0 = sorted(set(sec))[0]
    idx = np.where((sec == s0) & np.char.startswith(lab.astype(str), "F"))[0][20:]
    lab2 = lab.copy(); lab2[idx] = "Epi"
    A.obs["label"] = pd.Categorical(lab2)
    r2 = na.diagnose(A, "label", "^F", "section", counts_layer="counts", p=(0.6, 0.35), verbose=False,
                     min_targets=50)
    assert r2.overall["skipped_sections"] == [s0] and s0 not in set(r2.sections.section.astype(str))
    assert any("Skipped 1 section" in s for s in r2.interpretation)
    with pytest.raises(ValueError):
        na.diagnose(A, "label", "^F", "section", counts_layer="counts", p=(0.6,), verbose=False, min_targets=10 ** 6)
