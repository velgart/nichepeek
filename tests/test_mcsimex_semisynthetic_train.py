"""mcsimex --semisynthetic-train: the re-calling labeler learns the synthetic published labels on the
experiment counts (as when the protocol is run on semi-synthetic files), or the data's own labels on the full counts."""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
import anndata as ad
import scipy.sparse as sp

from nichepeek import io as _io
from nichepeek.api import _default_labeler
from nichepeek.config import Config
from nichepeek.mcsimex import run_mcsimex, semisynthetic
from nichepeek.simulate import make_anndata

KW = dict(counts_layer="counts", n_perm=2, n_splits=1, n_sim=1, n_pairs=2, verbose=False)


@pytest.fixture(scope="module")
def A():
    return make_anndata(n_sections=3, seed=4, n_cells=2500, depth=140, label_noise=0.0)


def _as_files(A):
    """The semi-synthetic data set written out to files."""
    cfg = Config(n_perm=2, n_splits=1)
    data = _io.load(A, "label", "section", counts_layer="counts", seed=cfg.seed)
    vc = pd.Series(data.label).value_counts()
    labels = np.array(sorted(vc[vc >= cfg.min_label_cells].index))
    keep = np.isin(data.label, labels)
    ref = _default_labeler(data, labels, keep, cfg)
    parts = []
    for u in pd.unique(data.section[keep]):
        ii = np.where(keep & (data.section == u))[0]
        YE, pub, tru = semisynthetic(u, data.counts[ii].toarray(), data.parent[ii][0], ref, 0.5, 1.0, cfg.seed)
        parts.append((ii, YE, pub, tru))
    ii = np.concatenate([p[0] for p in parts])
    obs = pd.DataFrame(dict(label=np.concatenate([p[2] for p in parts]), truth=np.concatenate([p[3] for p in parts]),
                            section=data.section[ii], x=data.xy[ii, 0], y=data.xy[ii, 1]))
    obs.index = obs.index.astype(str)
    B = ad.AnnData(X=sp.csr_matrix(np.vstack([p[1] for p in parts]).astype(np.float32)), obs=obs,
                   var=pd.DataFrame(index=data.genes))
    B.layers["counts"] = B.X.copy()
    return B


def test_synthetic_training_equals_the_files(A):
    R = run_mcsimex(A, "label", "^F", "section", semisynthetic_f=0.5, semisynthetic_train="synthetic", **KW)
    F = run_mcsimex(_as_files(A), "label", "^F", "section", truth_key="truth", **KW)
    R, F = R.set_index("section"), F.set_index("section").loc[R.section]
    for k in ["flip_native", "true_T", "first_level", "acc_c1_vs_truth_targets"]:
        np.testing.assert_allclose(R[k], F[k], rtol=1e-9, atol=1e-12, err_msg=k)


def test_reference_training_is_0_3_2_and_differs(A):
    R0 = run_mcsimex(A, "label", "^F", "section", semisynthetic_f=0.5, semisynthetic_train="reference", **KW)
    R1 = run_mcsimex(A, "label", "^F", "section", semisynthetic_f=0.5, **KW)          # default: synthetic
    assert R0.true_T.median() > 0 and R1.true_T.median() > 0
    assert not np.allclose(R0.flip_native, R1.flip_native)
    with pytest.raises(ValueError):
        run_mcsimex(A, "label", "^F", "section", semisynthetic_f=0.5, semisynthetic_train="other", **KW)
