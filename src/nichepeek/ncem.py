"""NCEM-style niche regression: ridge regression of size-normalized expression of
target cells on their own label (one-hot), their neighbor composition, or both; cross-validated R2 of
the biological variance.
    niche = R2(full) - R2(own label)   the niche effect beyond cell type (program analogue)
    type  = R2(full) - R2(neighbors)  the type effect beyond niche (identity analogue)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .ladder import _mn_var


def cv_r2(X, Y, folds, lam=1.0, pois=None):
    """Micro-averaged out-of-fold R2 of ridge regression; with pois, of the biological variance."""
    sse = sst = 0.0
    n = 0
    for k in np.unique(folds):
        te, tr = folds == k, folds != k
        xm, ym = X[tr].mean(0), Y[tr].mean(0)
        Xt, Yt = X[tr] - xm, Y[tr] - ym
        B = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ Yt)
        pred = (X[te] - xm) @ B + ym
        sse += ((Y[te] - pred) ** 2).sum()
        sst += ((Y[te] - ym) ** 2).sum()
        n += te.sum()
    if pois is not None:
        sse -= n * pois.sum()
        sst -= n * pois.sum()
    return 1 - sse / sst


def ncem(xy, own, Y, tgt, label_list, seed, nb_labels=None, k_nb=20, min_det=0.01, genes=None):
    """NCEM-style scores for the queried cells tgt; label roles separated as in ladder().
    genes: fixed gene indices (identical across arms); None = detected genes with biological variance.
    NB with labels called on the same counts (circular arms) the noise-corrected type effect can exceed 1:
    the correction assumes residuals keep the full sampling noise, which fitted labels remove."""
    own = np.asarray(own)
    nbl = own if nb_labels is None else np.asarray(nb_labels)
    it = np.where(np.asarray(tgt, bool))[0]
    if len(it) < 200:
        return {}
    nb = cKDTree(xy).query(xy[it], k=k_nb + 1)[1][:, 1:]
    codes = pd.Categorical(nbl, categories=list(label_list)).codes
    M = np.zeros((len(it), len(label_list)))
    for j in range(k_nb):
        np.add.at(M, (np.arange(len(it)), codes[nb[:, j]]), 1.0)
    Xnb = np.sqrt(M / k_nb)
    Xnb = Xnb[:, Xnb.std(0) > 0]
    Xown = pd.get_dummies(pd.Series(own[it])).to_numpy(float)
    Yc = np.asarray(Y[it], dtype=np.float64)
    t = np.maximum(Yc.sum(1), 1)
    sf = t / t.mean()
    fixed = genes is not None
    Yc = Yc[:, genes] if fixed else Yc[:, (Yc > 0).mean(0) >= min_det]
    Yl = Yc / sf[:, None]
    pois = _mn_var(Yc, t, sf)                     # multinomial: the size factor conditions on the total
    g = np.ones(Yc.shape[1], bool) if fixed else (Yl.var(0, ddof=1) - pois) > 0
    if g.sum() < 10:
        return {}
    Yl, pois = Yl[:, g], pois[g]
    folds = np.random.default_rng(seed).integers(0, 5, len(it))
    r_lab = cv_r2(Xown, Yl, folds, pois=pois)
    r_nb = cv_r2(Xnb, Yl, folds, pois=pois)
    r_full = cv_r2(np.column_stack([Xown, Xnb]), Yl, folds, pois=pois)
    return dict(ncem_full=r_full, ncem_niche=r_full - r_lab, ncem_type=r_full - r_nb)
