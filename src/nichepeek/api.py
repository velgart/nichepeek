"""diagnose(): the label-error diagnostic on a spatial AnnData object or .h5ad file."""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import io as _io
from .config import Config
from .labeling import CentroidLabeler, LogRegLabeler, as_labeler, norm_log
from .protocol import analyse_section
from .summarize import context_tests, interpret, overall, summarize


@dataclass
class Result:
    sections: pd.DataFrame
    summary: pd.DataFrame
    overall: dict
    interpretation: list
    config: Config
    meta: dict = field(default_factory=dict)

    def save(self, out_dir, report=True):
        os.makedirs(out_dir, exist_ok=True)
        self.sections.to_csv(os.path.join(out_dir, "sections.csv"), index=False)
        self.summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        json.dump(dict(overall=self.overall, interpretation=self.interpretation, config=asdict(self.config),
                       meta=self.meta), open(os.path.join(out_dir, "result.json"), "w"), indent=1, default=str)
        if report:
            self.report(os.path.join(out_dir, "report.pdf"))
        return out_dir

    def report(self, path):
        from .report import make_report
        return make_report(self, path)

    def __repr__(self):
        return "nichepeek.Result\n  " + "\n  ".join(self.interpretation)


def _default_labeler(data, labels, keep, cfg):
    lab = CentroidLabeler(labels)
    rng = np.random.default_rng([cfg.seed, 1])
    med_all = float(np.median(np.asarray(data.counts[keep].sum(1)).ravel()))
    for s in pd.unique(data.parent[keep]):
        for l in labels:
            idx = np.where(keep & (data.parent == s) & (data.label == l))[0]
            if len(idx) == 0:
                continue
            idx = np.sort(rng.choice(idx, min(len(idx), cfg.centroid_sample), replace=False))
            L = norm_log(data.counts[idx].toarray(), med_all)
            lab.add(s, l, L.sum(0), len(idx))
    return lab


def _run_one(args):
    sec, Y, xy, auth, parent, cond, labels, is_target, labeler, cfg, truth = args
    return analyse_section(sec, Y, xy, auth, parent, cond, labels, is_target, labeler, cfg, truth=truth)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _peak_memory_gb():
    """Peak resident memory of this process and of its finished worker processes (the largest of either), in GB."""
    import resource, sys
    unit = 1.0 if sys.platform == "darwin" else 1024.0          # macOS reports bytes, Linux kilobytes
    m = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return round(m * unit / 1e9, 2)


def diagnose(adata, label_key, target, section_key, spatial=("x", "y"), counts_layer="X",
             condition_key=None, sections=None, crop_cells=0, crops_per_section=2, labeler=None,
             config=None, out_dir=None, n_jobs=1, resume=True, verbose=True,
             exclude_genes=_io.DEFAULT_EXCLUDE, labels_file=None, truth_key=None, calibrate=None,
             calibration_tissue=None, **settings):
    """Run the label-error diagnostic.

    adata          AnnData object, or path to an .h5ad file (read row-selectively)
    label_key      obs column with the published cell-type labels
    target         the cells whose neighborhood share is analyzed: regex (searched in labels), list of labels,
                   or callable labels -> bool
    section_key    obs column with the section (sample) id
    spatial        (x_column, y_column) in obs, or an obsm key (e.g. 'spatial')
    counts_layer   'X', 'raw' or a layer name holding raw integer counts
    truth_key      optional obs column (or column of labels_file) with KNOWN true labels, for validation on
                   simulated or semi-synthetic data only: adds arms scoring the true labels on the same half and
                   split as every other arm, and the directly measured label-error bias of the re-called labels
                   (true_net_*, true_T_*, true_N_*). Not for real data, where the true labels are unknown.
    calibrate      None, or "default" / a path to a calibration table: adds a calibrated own-label bias to the
                   summary and report (see nichepeek.calibrate; needs a matching first thinning step)
    labels_file    optional table (csv/tsv/parquet, first column = cell ID) holding label_key, used instead of the
                   data file's obs; alternative label sets can then be analyzed without modifying the data file
    labeler        None or 'centroid' (nearest-centroid, leave-one-section-out), 'logreg' (logistic regression,
                   leave-one-section-out, trained on thinned counts), an object with predict(Y, section), or a
                   callable f(Y, section) -> labels. It is applied to half A and to thinned counts. The labeler's
                   class name is part of the cache key; give custom labelers distinct class names.
    config         nichepeek.Config; keyword settings (p=..., split=..., ncem=..., ...) override it
    out_dir        if given, per-section results are cached there (resumable) and the result is saved
    exclude_genes  regex of var names to drop (control probes, matched from the start of the name); to add
                   whole names, anchor them, e.g. for the Cadinu MERFISH blank aliases:
                   nichepeek.io.DEFAULT_EXCLUDE + "|^(?:Dclk1|Gdf1)$"   (unanchored "Gdf1" also drops Gdf10/11/15)
    n_jobs         sections analyzed in parallel (processes)
    Returns a Result (sections, summary, overall, interpretation)."""
    cfg = Config(**{**(asdict(config) if config else {}), **settings})
    cfg.p = tuple(float(x) for x in cfg.p)
    t0 = time.time()
    data = _io.load(adata, label_key, section_key, spatial=spatial, counts_layer=counts_layer,
                    condition_key=condition_key, sections=sections, crop_cells=crop_cells,
                    crops_per_section=crops_per_section, exclude_genes=exclude_genes, seed=cfg.seed,
                    labels_file=labels_file, truth_key=truth_key)
    vc = pd.Series(data.label).value_counts()
    labels = np.array(sorted(vc[vc >= cfg.min_label_cells].index))
    keep = np.isin(data.label, labels)
    is_target = _io.TargetMask(target, labels)
    if not is_target(labels).any():
        raise ValueError(f"target {target!r} matches none of the labels: {', '.join(labels)}")
    if isinstance(labeler, str):
        if labeler == "logreg":
            lab = LogRegLabeler.from_data(data, labels, keep, seed=cfg.seed)
        elif labeler == "centroid":
            lab = _default_labeler(data, labels, keep, cfg)
        else:
            raise ValueError(f"unknown labeler {labeler!r}: 'centroid', 'logreg', an object or a callable")
    else:
        lab = as_labeler(labeler) or _default_labeler(data, labels, keep, cfg)
    units = list(pd.unique(data.section[keep]))
    skipped = []                                      # too few published targets for a stable ladder
    if cfg.min_targets:
        for u in list(units):
            nt = int(is_target(data.label[keep & (data.section == u)]).sum())
            if nt < cfg.min_targets:
                skipped.append(u)
                units.remove(u)
                if verbose:
                    print(f"  skipping {u}: {nt} target cells (< min_targets {cfg.min_targets})", flush=True)
        if not units:
            raise ValueError(f"no section has at least {cfg.min_targets} target cells (min_targets)")
    if verbose:
        print(f"{keep.sum()} cells, {len(data.genes)} genes, {len(labels)} labels "
              f"({int(is_target(labels).sum())} target labels), {len(units)} sections [{time.time() - t0:.0f} s]",
              flush=True)
    # the cache key covers the analysis settings AND everything that changes a section's input, so a rerun
    # with, e.g., other gene exclusions or crop sizes never reuses stale sections
    import hashlib
    lf = None
    if labels_file:                                   # the label table's content is part of the cache key
        with open(labels_file, "rb") as fh:
            lf = [str(labels_file), hashlib.sha1(fh.read()).hexdigest()]
    inp = dict(src=adata if isinstance(adata, str) else "memory", label_key=label_key, target=str(target),
               labels_file=lf,
               section_key=section_key, spatial=spatial, counts_layer=counts_layer,
               crop_cells=crop_cells, crops_per_section=crops_per_section, exclude_genes=exclude_genes,
               n_genes=len(data.genes), labeler=type(lab).__name__)
    if truth_key:
        # added only when given, so runs without it keep their key. A semi-synthetic file can be rebuilt
        # under the same name, so its size and modification time enter the key too.
        inp["truth_key"] = truth_key
        if isinstance(adata, str):
            st = os.stat(adata)
            inp["src_stat"] = [st.st_size, int(st.st_mtime)]
    inp = json.dumps(inp, sort_keys=True, default=str)
    ikey = hashlib.sha1(inp.encode()).hexdigest()[:8]
    cache = os.path.join(out_dir, "sections", f"{cfg.key()}_{ikey}") if out_dir else None
    if cache:
        os.makedirs(cache, exist_ok=True)

    def job(u):
        ii = np.where(keep & (data.section == u))[0]
        return (u, data.counts[ii].toarray(), data.xy[ii], data.label[ii], data.parent[ii][0],
                data.condition[ii][0], labels, is_target, lab, cfg,
                data.truth[ii] if data.truth is not None else None)

    recs, ctxs, todo = {}, {}, []
    for u in units:
        f = os.path.join(cache, _safe(u) + ".npz") if cache else None
        if f and resume and os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            recs[u], ctxs[u] = z["rec"].item(), z["ctx"].item()
        else:
            todo.append(u)

    def done(u, rec, ctx):
        recs[u], ctxs[u] = rec, ctx
        if cache:
            np.savez(os.path.join(cache, _safe(u) + ".npz"), rec=np.array(rec, dtype=object),
                     ctx=np.array(ctx, dtype=object))
        if verbose:
            print(f"  {u} ({rec['n_cells']} cells, median {rec['median_counts']:.0f} counts): "
                  f"native flip {rec['flip_native']:.2f} | "
                  + " | ".join(f"p={p:g}: L_P {rec[f'L_program_{p}']:+.3f}"
                               + (f" (T {rec[f'L_T_program_{p}']:+.3f})" if cfg.channels else "")
                               for p in cfg.p) + f" [{time.time() - t0:.0f} s]", flush=True)

    if n_jobs > 1 and len(todo) > 1:
        # at most n_jobs sections in flight: each job carries a dense copy of its section's counts
        with ProcessPoolExecutor(n_jobs) as ex:
            queue, running = list(todo), {}
            while queue or running:
                while queue and len(running) < n_jobs:
                    u = queue.pop(0)
                    running[ex.submit(_run_one, job(u))] = u
                fin, _ = wait(running, return_when=FIRST_COMPLETED)
                for fu in fin:
                    done(running.pop(fu), *fu.result())
    else:
        for u in todo:
            done(u, *_run_one(job(u)))

    R = pd.DataFrame([recs[u] for u in units])
    ctx = context_tests([ctxs[u] for u in units], cfg.p, seed=cfg.seed) if cfg.context_tests else None
    S = summarize(R, cfg.p, ctx)
    o = overall(R, cfg.p, min_depth=getattr(cfg, "min_depth", 10.0))
    o["n_target_labels"] = int(is_target(labels).sum())
    o["min_targets"] = int(cfg.min_targets)
    o["skipped_sections"] = [str(u) for u in skipped]
    if calibrate:
        from .calibrate import calibrate as _cal
        try:
            o.update(_cal(o, None if calibrate == "default" else calibrate, calibration_tissue))
        except (FileNotFoundError, ValueError) as err:
            o["calibration_note"] = str(err)
    peak = _peak_memory_gb()
    o["seconds"], o["peak_memory_gb"] = round(time.time() - t0), peak
    res = Result(R, S, o, interpret(o, S), cfg,
                 meta=dict(version=__import__("nichepeek").__version__, labeler=type(lab).__name__, truth_key=truth_key, n_cells=int(keep.sum()), n_genes=len(data.genes), labels=list(labels),
                           target_labels=is_target.target_labels, seconds=round(time.time() - t0),
                           peak_memory_gb=peak))
    if out_dir:
        res.save(out_dir)
    if verbose:
        print("\n" + "\n".join(res.interpretation))
    return res
