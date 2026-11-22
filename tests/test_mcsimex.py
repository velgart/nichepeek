"""Misclassification SIMEX and pair sensitivity of the own-label channel (nichepeek mcsimex)."""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
from nichepeek.mcsimex import confusion, draw, matrix_power, run_mcsimex, summarize_mcsimex
from nichepeek.simulate import make_anndata


def test_matrix_power_is_stochastic_and_composes():
    Pi = np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.02, 0.05, 0.93]])
    for lam in (0.5, 1.0, 1.5, 2.0):
        M, neg = matrix_power(Pi, lam)
        np.testing.assert_allclose(M.sum(1), 1.0)
        assert (M >= 0).all()
    H, _ = matrix_power(Pi, 0.5)
    np.testing.assert_allclose(H @ H, Pi, atol=1e-6)
    np.testing.assert_allclose(matrix_power(Pi, 2.0)[0], Pi @ Pi)


def test_draw_follows_the_confusion_matrix():
    rng = np.random.default_rng(0)
    codes = np.repeat([0, 1], 20000)
    M = np.array([[0.8, 0.2], [0.1, 0.9]])
    out = draw(codes, np.arange(len(codes)), M, rng)
    Pi, _ = confusion(codes, out, None, 2)
    np.testing.assert_allclose(Pi, M, atol=0.01)
    assert (draw(codes, np.arange(0), M, rng) == codes).all()      # cells outside idx are untouched


@pytest.fixture(scope="module")
def run():
    A = make_anndata(n_sections=3, seed=2, n_cells=2500, depth=70, label_noise=0.1)
    return run_mcsimex(A, "label", "^F", "section", counts_layer="counts", truth_key="true_label", n_perm=2,
                       n_splits=1, n_sim=1, n_pairs=3, verbose=False)


def test_mcsimex_runs_and_reports(run):
    R = run
    for k in ["true_T", "first_level", "mcsimex_lin", "mcsimex_quad", "oracle_lin", "oracle_quad", "pairsum",
              "pairsum_true", "pair_coverage", "pair0", "pair0_slope", "pair0_merge"]:
        assert k in R
    S = summarize_mcsimex(R)
    assert "ratio_to_true" in S
    assert R.true_T.median() > 0                     # the own-label bias is positive in simulations


def test_mcsimex_shares_split_and_first_step_with_the_protocol(run):
    # with the same seed, p0 = the protocol's first level and one split, c1, flips and the first thinning
    # are those of diagnose(): same native flip rate and same own-label effect at p0
    import nichepeek as na
    A = make_anndata(n_sections=3, seed=2, n_cells=2500, depth=70, label_noise=0.1)
    d = na.diagnose(A, "label", "^F", "section", counts_layer="counts", p=(0.8, 0.5), n_perm=2, verbose=False,
                    context_tests=False, membership=False)
    D = d.sections.set_index("section")
    R = run.set_index("section")
    np.testing.assert_allclose(R.flip_native, D.loc[R.index, "flip_native"], atol=1e-12)
    np.testing.assert_allclose(R.L_T_p0, D.loc[R.index, "L_T_program_0.8"], atol=1e-12)


def test_group_breakdown(run):
    G = run.attrs["groups"]
    assert set(G.section) == set(run.section)
    for col in ["n", "share_n", "mix_dev", "flip", "id_pub", "id_rc", "resp", "score", "id_true", "loss_true"]:
        assert col in G
    np.testing.assert_allclose(G.groupby("section").share_n.sum(), 1.0)
    assert (G.score >= 0).all()
    assert G.groupby("section").loss_true.sum().median() > 0      # true labels carry more identity
    assert "grp_rho_score_loss" in run


def test_semisynthetic_mode(tmp_path):
    # the semi-synthetic split built on the fly; truth = labels of the truth share, published = labels of
    # the experiment share on its own counts; the measured own-label bias is positive, as on the paper's files
    A = make_anndata(n_sections=3, seed=4, n_cells=2500, depth=140, label_noise=0.0)
    R = run_mcsimex(A, "label", "^F", "section", counts_layer="counts", semisynthetic_f=0.5, n_perm=2,
                    n_splits=1, n_sim=1, n_pairs=2, verbose=False)
    assert R.true_T.median() > 0
    assert (R.acc_c1_vs_truth_targets < 1).all()
    G = R.attrs["groups"]
    assert "loss_true" in G and len(G) == 3 * 12
    with pytest.raises(ValueError):
        run_mcsimex(A, "label", "^F", "section", counts_layer="counts", semisynthetic_f=0.5,
                    truth_key="true_label", verbose=False)
