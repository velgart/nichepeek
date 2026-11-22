import numpy as np
import pytest

from nichepeek.simulate import make_anndata


@pytest.fixture(scope="session")
def adata():
    # 3 small sections with known truth; published labels = true labels, depth and thinning in the regime
    # validated in the paper's simulations (at < ~10 counts per cell the channel signs are not reliable)
    return make_anndata(n_sections=3, seed=1, label_noise=0.0, n_cells=4000, depth=70)


@pytest.fixture(scope="session")
def result(adata):
    import nichepeek as na
    return na.diagnose(adata, "label", "^F", "section", counts_layer="counts", p=(0.6, 0.35), verbose=False)
