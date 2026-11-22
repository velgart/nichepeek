"""Native flip rates at full and at half depth, and overlap between crops.

The protocol's `flip_native` is measured on half A of the counts (labels are called there), so it describes
labels called at half depth. Published labels were called at full depth. This module measures both, with the
same labeler and the same sections/crops as a diagnostic run, without running the ladder:

  flip_full_pub / flip_full_rc   labels re-called on Poisson(Y) vs on Y (full counts), among published /
                                 re-called targets
  flip_half_rc                   the protocol's definition (Poisson(YA) vs YA, re-called targets on YA);
                                 should reproduce sections.csv flip_native up to resampling noise
  crop_overlap                   fraction of a crop's cells that also belong to another crop of the same
                                 physical section (0 when sections are not cropped)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io as _io
from .api import _default_labeler
from .config import Config
from .labeling import as_labeler
from .protocol import section_seed


def flip_rates(adata, label_key, target, section_key, spatial=("x", "y"), counts_layer="X", sections=None,
               crop_cells=0, crops_per_section=2, labeler=None, exclude_genes=_io.DEFAULT_EXCLUDE, labels_file=None,
               n_boot=5, seed=0, verbose=True, **settings):
    cfg = Config(seed=seed, **settings)
    data = _io.load(adata, label_key, section_key, spatial=spatial, counts_layer=counts_layer, sections=sections,
                    crop_cells=crop_cells, crops_per_section=crops_per_section, exclude_genes=exclude_genes,
                    seed=cfg.seed, labels_file=labels_file)
    vc = pd.Series(data.label).value_counts()
    labels = np.array(sorted(vc[vc >= cfg.min_label_cells].index))
    keep = np.isin(data.label, labels)
    is_target = _io.TargetMask(target, labels)
    lab = as_labeler(labeler) or _default_labeler(data, labels, keep, cfg)
    rows = []
    units = list(pd.unique(data.section[keep]))
    # crop overlap: cells are identified by their coordinates within a physical section
    key = pd.Series([f"{p}|{x:.6g}|{y:.6g}" for p, (x, y) in zip(data.parent, data.xy)])
    for u in units:
        ii = np.where(keep & (data.section == u))[0]
        Y = np.asarray(data.counts[ii].toarray(), dtype=np.int64)
        parent = data.parent[ii][0]
        auth = data.label[ii]
        tg_pub = is_target(auth)
        rng = np.random.default_rng(section_seed(seed, u) + [7])
        med_full = float(np.median(Y.sum(1)))
        c_full = lab.predict(Y, parent, med_full)
        fl = np.zeros(len(ii))
        for _ in range(n_boot):
            fl += lab.predict(rng.poisson(Y), parent, med_full) != c_full
        fl /= n_boot
        tg_rc = is_target(c_full)
        YA = rng.binomial(Y, 0.5)
        med_half = float(np.median(YA.sum(1)))
        c_half = lab.predict(YA, parent, med_half)
        fh = np.zeros(len(ii))
        for _ in range(n_boot):
            fh += lab.predict(rng.poisson(YA), parent, med_half) != c_half
        fh /= n_boot
        others = [v for v in units if v != u and np.any(data.parent[data.section == v] == parent)]
        ov = 0.0
        if others:
            mine = set(key[ii])
            other = set(key[np.where(np.isin(data.section, others))[0]])
            ov = len(mine & other) / max(len(mine), 1)
        rec = dict(section=u, parent=parent, n_cells=len(ii), n_target=int(tg_pub.sum()),
                   median_counts=med_full,
                   flip_full_pub=float(fl[tg_pub].mean()) if tg_pub.any() else np.nan,
                   flip_full_rc=float(fl[tg_rc].mean()) if tg_rc.any() else np.nan,
                   flip_half_rc=float(fh[is_target(c_half)].mean()) if is_target(c_half).any() else np.nan,
                   acc_full_vs_published=float((c_full == auth).mean()),
                   acc_full_targets=float((c_full[tg_pub] == auth[tg_pub]).mean()) if tg_pub.any() else np.nan,
                   crop_overlap=float(ov))
        rows.append(rec)
        if verbose:
            print(f"  {u}: flip full {rec['flip_full_pub']:.3f} (pub. targets) / {rec['flip_full_rc']:.3f}, "
                  f"half {rec['flip_half_rc']:.3f}; crop overlap {ov:.2f}", flush=True)
    return pd.DataFrame(rows)
