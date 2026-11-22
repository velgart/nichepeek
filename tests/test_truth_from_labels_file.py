"""The truth column of a tab-separated or parquet labels file; the markers command is not offered."""
import numpy as np
import pandas as pd
import pytest

from nichepeek import io
from nichepeek.cli import main
from nichepeek.simulate import make_anndata


@pytest.mark.parametrize("ext", ["tsv", "csv", "parquet"])
def test_truth_key_from_labels_file(tmp_path, ext):
    A = make_anndata(n_sections=2, seed=3, n_cells=600, depth=40, label_noise=0.1)
    L = pd.DataFrame({"pub": A.obs["label"].astype(str), "truth": A.obs["true_label"].astype(str)}, index=A.obs.index)
    f = tmp_path / f"labels.{ext}"
    if ext == "parquet":
        pytest.importorskip("pyarrow")
        L.to_parquet(f)
    else:
        L.to_csv(f, sep="\t" if ext == "tsv" else ",")
    D = io.load(A, "pub", "section", counts_layer="counts", labels_file=str(f), truth_key="truth")
    assert np.array_equal(D.truth, A.obs["true_label"].astype(str).to_numpy())
    assert np.array_equal(D.label, A.obs["label"].astype(str).to_numpy())


def test_markers_command_removed():
    with pytest.raises(SystemExit):
        main(["markers", "calls.csv", "--out", "x"])
