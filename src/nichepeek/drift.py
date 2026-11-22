"""Label drift and confusion structure across thinning levels (no ladder).

For each section, labels are re-called on half A of the counts (c1), on Poisson resamples of half A (native
flips) and on half A thinned to each level p (added flips), exactly as in the protocol. Reported per section and
level, among target cells (published labels):

  largest_share, entropy      class-frequency distribution of the re-called labels (does it drift with depth?)
  dissim_native, dissim_added mean dissimilarity (1 - correlation of the published labels' mean log profiles)
                              between the label a cell leaves and the label it lands on, for native and added flips
  confusion_corr              correlation between the native and added flip-transition frequencies (off-diagonal):
                              do thinning-induced errors have the same confusion structure as native errors?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io as _io
from .api import _default_labeler
from .config import Config
from .labeling import LogRegLabeler, as_labeler, norm_log
from .protocol import section_seed


def _entropy(lab):
    p = pd.Series(lab).value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum())


def label_drift(adata, label_key, target, section_key, spatial=("x", "y"), counts_layer="X", sections=None,
                crop_cells=0, crops_per_section=2, labeler=None, exclude_genes=_io.DEFAULT_EXCLUDE, labels_file=None,
                p=(0.5, 0.25, 0.12), n_boot=5, seed=0, verbose=True, **settings):
    cfg = Config(seed=seed, **settings)
    data = _io.load(adata, label_key, section_key, spatial=spatial, counts_layer=counts_layer, sections=sections,
                    crop_cells=crop_cells, crops_per_section=crops_per_section, exclude_genes=exclude_genes,
                    seed=cfg.seed, labels_file=labels_file)
    vc = pd.Series(data.label).value_counts()
    labels = np.array(sorted(vc[vc >= cfg.min_label_cells].index))
    keep = np.isin(data.label, labels)
    is_target = _io.TargetMask(target, labels)
    if isinstance(labeler, str):                        # same names as diagnose(labeler=...)
        if labeler == "logreg":
            lab = LogRegLabeler.from_data(data, labels, keep, seed=cfg.seed)
        elif labeler == "centroid":
            lab = _default_labeler(data, labels, keep, cfg)
        else:
            raise ValueError(f"unknown labeler {labeler!r}: 'centroid', 'logreg', an object or a callable")
    else:
        lab = as_labeler(labeler) or _default_labeler(data, labels, keep, cfg)
    rows = []
    for u in pd.unique(data.section[keep]):
        ii = np.where(keep & (data.section == u))[0]
        Y = np.asarray(data.counts[ii].toarray(), dtype=np.int32)
        parent, auth = data.parent[ii][0], data.label[ii]
        tg = is_target(auth)
        if tg.sum() < 20:
            continue
        rng = np.random.default_rng(section_seed(seed, u) + [11])
        L = norm_log(Y, float(np.median(Y.sum(1))))
        prof = pd.DataFrame(L).groupby(auth).mean()
        D = 1 - np.corrcoef(prof.to_numpy())
        idx = {l: i for i, l in enumerate(prof.index)}
        dis = lambda a, b: np.array([D[idx[x], idx[y]] if x in idx and y in idx else np.nan for x, y in zip(a, b)])
        YA = rng.binomial(Y, 0.5).astype(np.int32)
        med = float(np.median(YA.sum(1)))
        c1 = lab.predict(YA, parent, med)
        nat_from, nat_to = [], []
        for _ in range(n_boot):
            r = lab.predict(rng.poisson(YA), parent, med)
            m = tg & (r != c1)
            nat_from.append(c1[m]); nat_to.append(r[m])
        nat_from, nat_to = np.concatenate(nat_from), np.concatenate(nat_to)
        names = sorted(set(c1[tg]) | set(nat_to))
        def trans(f, t):
            M = pd.crosstab(pd.Series(f, name="f"), pd.Series(t, name="t")).reindex(index=names, columns=names,
                                                                                   fill_value=0).to_numpy(float).copy()
            np.fill_diagonal(M, 0)
            return M / max(M.sum(), 1)
        Mn = trans(nat_from, nat_to)
        base = dict(section=u, parent=parent, n_target=int(tg.sum()), largest_share_c1=float(
            pd.Series(c1[tg]).value_counts(normalize=True).iloc[0]), entropy_c1=_entropy(c1[tg]),
            dissim_native=float(np.nanmean(dis(nat_from, nat_to))) if len(nat_from) else np.nan)
        for q in p:
            cp = lab.predict(rng.binomial(YA, q).astype(np.int32), parent, med)
            m = tg & (cp != c1)
            Ma = trans(c1[m], cp[m])
            off = ~np.eye(len(names), dtype=bool)
            cc = float(np.corrcoef(Mn[off], Ma[off])[0, 1]) if Mn[off].std() > 0 and Ma[off].std() > 0 else np.nan
            rows.append(dict(base, p=q, added_flip=float(m[tg].mean()),
                             largest_share=float(pd.Series(cp[tg]).value_counts(normalize=True).iloc[0]),
                             entropy=_entropy(cp[tg]),
                             dissim_added=float(np.nanmean(dis(c1[m], cp[m]))) if m.any() else np.nan,
                             confusion_corr=cc))
        if verbose:
            r = rows[-1]
            print(f"  {u}: entropy {base['entropy_c1']:.2f} -> {r['entropy']:.2f} at p={q:g}; dissimilarity "
                  f"native {base['dissim_native']:.3f}, added {r['dissim_added']:.3f}; confusion r {r['confusion_corr']:.2f}",
                  flush=True)
    return pd.DataFrame(rows)
