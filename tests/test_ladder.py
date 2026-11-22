import numpy as np
import pytest

from nichepeek.ladder import ladder, r2_micro, sampling_variance
from nichepeek.simulate import TGT, section
from reference_ladder import ladder_ref


@pytest.fixture(scope="module")
def sec():
    rng = np.random.default_rng(3)
    xy, lab, Y = section(rng, n_cells=3000, depth=50)
    noisy = lab.copy()
    swap = {"FA": "FA2", "FA2": "FA", "FB": "FB", "FB2": "FB2"}
    m = np.isin(lab, TGT) & (rng.random(len(lab)) < 0.3)
    noisy[m] = np.array([{"FA": "FB", "FB": "FA", "FA2": "FB2", "FB2": "FA2"}[x] for x in lab[m]])
    return xy, lab, noisy, Y


def is_t(l):
    return np.isin(l, TGT)


def test_reproduces_paper_ladder(sec):
    xy, lab, noisy, Y = sec
    for labels in (lab, noisy):
        ref = ladder_ref(xy, labels, Y, 7, is_t)
        new = ladder(xy, labels, Y, 7, is_t(labels))
        for k in ref:
            assert new[k] == pytest.approx(ref[k], abs=1e-12), k


def test_accounting_closes(sec):
    xy, lab, noisy, Y = sec
    o = ladder(xy, noisy, Y, 7, is_t(lab))
    assert o["program"] + o["identity"] == pytest.approx(o["S0"] - o["S3"], abs=1e-12)


def test_own_label_channel_leaves_S0_unchanged(sec):
    xy, lab, noisy, Y = sec
    tg = is_t(lab)
    base = ladder(xy, lab, Y, 7, tg)
    own_err = ladder(xy, noisy, Y, 7, tg, nb_labels=lab)      # error only in the targets' own labels
    assert own_err["S0"] == base["S0"]                        # exactly: S0 never sees a cell's own label
    nb_err = ladder(xy, lab, Y, 7, tg, nb_labels=noisy)       # error only in the neighbor role
    assert nb_err["S1"] - nb_err["S3"] != base["S1"] - base["S3"] or nb_err["S0"] != base["S0"]


def test_multinomial_variance_is_smaller(sec):
    _, _, _, Y = sec
    Y = Y.astype(float)
    sf = Y.sum(1) / Y.sum(1).mean()
    assert np.all(sampling_variance(Y, sf, "multinomial") <= sampling_variance(Y, sf, "poisson") + 1e-12)


def test_true_labels_have_identity(sec):
    xy, lab, noisy, Y = sec
    assert ladder(xy, lab, Y, 7, is_t(lab))["identity"] > 0.02
