"""Command line: nichepeek run | simulate | report | calibrate | flips | drift | mcsimex."""
from __future__ import annotations

import argparse
import json
import sys


def _floats(s):
    return tuple(float(x) for x in s.split(","))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nichepeek", description="Label-error diagnostic for spatial niche analyses")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the diagnostic on an .h5ad file")
    r.add_argument("h5ad")
    r.add_argument("--label-key", required=True, help="obs column with published cell-type labels")
    r.add_argument("--target", required=True, help="regex selecting the target labels (searched in labels)")
    r.add_argument("--section-key", required=True)
    r.add_argument("--x", default="x"); r.add_argument("--y", default="y")
    r.add_argument("--spatial-key", default=None, help="obsm key with coordinates (instead of --x/--y)")
    r.add_argument("--counts-layer", default="X", help="'X', 'raw' or a layer name with raw integer counts")
    r.add_argument("--condition-key", default=None)
    r.add_argument("--sections", default=None, help="comma list of sections to analyze")
    r.add_argument("--crop-cells", type=int, default=0, help="analyze discs of this many cells per section")
    r.add_argument("--crops-per-section", type=int, default=2)
    r.add_argument("--p", type=_floats, default=(0.5, 0.25, 0.12), help="thinning levels")
    r.add_argument("--no-split", action="store_true", help="naive design: labels and ladder on the same counts")
    r.add_argument("--reselect-targets", action="store_true", help="targets follow each arm's labels (naive design)")
    r.add_argument("--no-channels", action="store_true")
    r.add_argument("--no-membership", action="store_true")
    r.add_argument("--ncem", action="store_true", help="also score the NCEM-style niche regression")
    r.add_argument("--ncem-genes", type=int, default=200, help="genes for the NCEM model (top N by mean in targets)")
    r.add_argument("--noise", choices=["poisson", "multinomial"], default="multinomial")
    r.add_argument("--exclude-genes", default=None, help="regex of var names to drop (default: common control probes)")
    r.add_argument("--n-splits", type=int, default=1,
                   help="independent random splits averaged per section (reduces split-to-split noise)")
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--no-resume", action="store_true")

    s = sub.add_parser("simulate", help="write a synthetic dataset with known truth (for trying the tool)")
    s.add_argument("out"); s.add_argument("--n-sections", type=int, default=4)
    s.add_argument("--n-cells", type=int, default=6000); s.add_argument("--depth", type=float, default=70)
    s.add_argument("--label-noise", type=float, default=0.0); s.add_argument("--seed", type=int, default=0)

    r.add_argument("--n-perm", type=int, default=1, help="within-label permutations averaged at the S1 rung")
    r.add_argument("--k", type=int, default=20, help="nearest neighbors defining a cell's neighborhood")
    r.add_argument("--min-targets", type=int, default=50,
                   help="skip sections (crops) with fewer published target cells (0 = keep all)")
    r.add_argument("--n-groups", type=int, default=12, help="k-means neighborhood groups")
    r.add_argument("--no-clip", action="store_true", help="unclipped R2 (negative per-gene residual variances kept)")
    r.add_argument("--labels-file", default=None, help="csv/tsv/parquet with cell IDs and label_key; data file untouched")
    r.add_argument("--calibrate", nargs="?", const="default", default=None,
                   help="add a calibrated own-label bias (optional: path to a calibration table)")
    r.add_argument("--calibration-tissue", default=None, help="prefer calibration rows of this tissue")
    r.add_argument("--truth-key", default=None,
                   help="VALIDATION ONLY: column with known true labels (simulated/semi-synthetic data); adds arms "
                        "measuring the label-error bias directly")
    r.add_argument("--labeler", default="centroid", choices=["centroid", "logreg"],
                   help="labeler that re-calls labels (default nearest-centroid)")

    f = sub.add_parser("flips", help="native flip rates at full and half depth, and crop overlap (no ladder)")
    for a_, kw in [("h5ad", {}), ("--label-key", dict(required=True)), ("--target", dict(required=True)),
                   ("--section-key", dict(required=True)), ("--x", dict(default="x")), ("--y", dict(default="y")),
                   ("--spatial-key", dict(default=None)), ("--counts-layer", dict(default="X")),
                   ("--sections", dict(default=None)), ("--crop-cells", dict(type=int, default=0)),
                   ("--crops-per-section", dict(type=int, default=2)), ("--exclude-genes", dict(default=None)),
                   ("--n-boot", dict(type=int, default=5)), ("--seed", dict(type=int, default=0)),
                   ("--out", dict(required=True, help="output csv")), ("--labels-file", dict(default=None))]:
        f.add_argument(a_, **kw)
    for a_ in ["--condition-key", "--p", "--noise", "--n-splits", "--n-jobs", "--ncem-genes", "--n-perm", "--labeler",
               "--k", "--n-groups", "--truth-key"]:
        f.add_argument(a_, default=None, help=argparse.SUPPRESS)          # accepted and ignored, so a `run`
    for a_ in ["--no-split", "--reselect-targets", "--no-channels", "--no-membership", "--ncem", "--no-resume",
               "--no-clip"]:
        f.add_argument(a_, action="store_true", help=argparse.SUPPRESS)    # line can be reused unchanged

    d = sub.add_parser("drift", help="label drift and confusion structure across thinning levels (no ladder)")
    for a_, kw in [("h5ad", {}), ("--label-key", dict(required=True)), ("--target", dict(required=True)),
                   ("--section-key", dict(required=True)), ("--x", dict(default="x")), ("--y", dict(default="y")),
                   ("--spatial-key", dict(default=None)), ("--counts-layer", dict(default="X")),
                   ("--sections", dict(default=None)), ("--crop-cells", dict(type=int, default=0)),
                   ("--crops-per-section", dict(type=int, default=2)), ("--exclude-genes", dict(default=None)),
                   ("--p", dict(type=_floats, default=(0.5, 0.25, 0.12))), ("--seed", dict(type=int, default=0)),
                   ("--labeler", dict(default="centroid")), ("--out", dict(required=True, help="output csv")), ("--labels-file", dict(default=None))]:
        d.add_argument(a_, **kw)
    for a_ in ["--condition-key", "--noise", "--n-splits", "--n-jobs", "--ncem-genes", "--n-perm", "--k", "--n-groups",
               "--truth-key"]:
        d.add_argument(a_, default=None, help=argparse.SUPPRESS)
    for a_ in ["--no-split", "--reselect-targets", "--no-channels", "--no-membership", "--ncem", "--no-resume",
               "--no-clip"]:
        d.add_argument(a_, action="store_true", help=argparse.SUPPRESS)

    m = sub.add_parser("mcsimex", help="misclassification SIMEX and pair sensitivity of the own-label channel "
                                      "(experimental)")
    for a_, kw in [("h5ad", {}), ("--label-key", dict(required=True)), ("--target", dict(required=True)),
                   ("--section-key", dict(required=True)), ("--x", dict(default="x")), ("--y", dict(default="y")),
                   ("--spatial-key", dict(default=None)), ("--counts-layer", dict(default="X")),
                   ("--condition-key", dict(default=None)),
                   ("--sections", dict(default=None)), ("--crop-cells", dict(type=int, default=0)),
                   ("--crops-per-section", dict(type=int, default=2)), ("--exclude-genes", dict(default=None)),
                   ("--labels-file", dict(default=None)), ("--truth-key", dict(default=None)),
                   ("--labeler", dict(default="centroid", choices=["centroid", "logreg"])),
                   ("--p0", dict(type=float, default=0.8, help="first thinning step (first-level estimate)")),
                   ("--lambdas", dict(type=_floats, default=(0.5, 1.0, 1.5, 2.0), help="MC-SIMEX error multiples")),
                   ("--n-sim", dict(type=int, default=2, help="draws per lambda")),
                   ("--n-pairs", dict(type=int, default=6, help="most frequent native confusions tested")),
                   ("--pair-frac", dict(type=float, default=0.2, help="share of a label's targets relabeled")),
                   ("--n-splits", dict(type=int, default=3)), ("--n-perm", dict(type=int, default=20)),
                   ("--k", dict(type=int, default=20)), ("--n-groups", dict(type=int, default=12)),
                   ("--min-targets", dict(type=int, default=50)),
                   ("--noise", dict(choices=["poisson", "multinomial"], default="multinomial")),
                   ("--semisynthetic", dict(type=float, default=None,
                                            help="validate on a semi-synthetic split built on the fly: truth share F")),
                   ("--semisynthetic-q", dict(type=float, default=1.0, help="experiment share thinned to Q")),
                   ("--semisynthetic-train", dict(choices=["synthetic", "reference"], default="synthetic",
                                                  help="labels that train the re-calling labeler: the synthetic "
                                                       "published labels on the experiment counts (as from files; "
                                                       "the default) or the data's own labels on the full counts")),
                   ("--n-jobs", dict(type=int, default=1)), ("--seed", dict(type=int, default=0)),
                   ("--out", dict(required=True, help="output directory"))]:
        m.add_argument(a_, **kw)

    q = sub.add_parser("report", help="re-draw the report of a finished run (summaries re-derived from sections.csv)")
    q.add_argument("out")
    q.add_argument("--calibrate", nargs="?", const="default", default=None,
                   help="add a calibrated own-label bias (optional: path to a calibration table)")
    q.add_argument("--calibration-tissue", default=None, help="prefer calibration rows of this tissue")

    c = sub.add_parser("calibrate", help="calibrated own-label bias of a finished run (needs a calibration table)")
    c.add_argument("out", help="result directory of a run")
    c.add_argument("--table", default=None, help="calibration table (default: the one shipped with the package)")
    c.add_argument("--tissue", default=None, help="prefer calibration rows of this tissue (e.g. xenium)")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .api import diagnose
        diagnose(a.h5ad, a.label_key, a.target, a.section_key,
                 spatial=a.spatial_key or (a.x, a.y), counts_layer=a.counts_layer, condition_key=a.condition_key,
                 sections=a.sections.split(",") if a.sections else None, crop_cells=a.crop_cells,
                 crops_per_section=a.crops_per_section, out_dir=a.out,
                 **({"exclude_genes": a.exclude_genes} if a.exclude_genes else {}), n_jobs=a.n_jobs, resume=not a.no_resume,
                 p=a.p, split=not a.no_split, fixed_targets=not a.reselect_targets, channels=not a.no_channels,
                 membership=not a.no_membership, ncem=a.ncem, ncem_genes=a.ncem_genes, noise=a.noise,
                 n_splits=a.n_splits, n_perm=a.n_perm, labeler=a.labeler, k=a.k, n_groups=a.n_groups,
                 labels_file=a.labels_file, truth_key=a.truth_key, calibrate=a.calibrate,
                 calibration_tissue=a.calibration_tissue,
                 clip=not a.no_clip, min_targets=a.min_targets,
                 seed=a.seed)
    elif a.cmd == "mcsimex":
        import os
        import pandas as pd
        from .mcsimex import run_mcsimex, summarize_mcsimex
        os.makedirs(a.out, exist_ok=True)
        R = run_mcsimex(a.h5ad, a.label_key, a.target, a.section_key, spatial=a.spatial_key or (a.x, a.y),
                        counts_layer=a.counts_layer, condition_key=a.condition_key,
                        sections=a.sections.split(",") if a.sections else None, crop_cells=a.crop_cells,
                        crops_per_section=a.crops_per_section, labeler=a.labeler, truth_key=a.truth_key,
                        labels_file=a.labels_file, p0=a.p0, lambdas=a.lambdas, n_sim=a.n_sim, n_pairs=a.n_pairs,
                        pair_frac=a.pair_frac, n_jobs=a.n_jobs, out_csv=os.path.join(a.out, "sections.csv"),
                        semisynthetic_f=a.semisynthetic, semisynthetic_q=a.semisynthetic_q,
                        semisynthetic_train=a.semisynthetic_train,
                        n_splits=a.n_splits, n_perm=a.n_perm, k=a.k, n_groups=a.n_groups, noise=a.noise,
                        min_targets=a.min_targets, seed=a.seed,
                        **({"exclude_genes": a.exclude_genes} if a.exclude_genes else {}))
        S = summarize_mcsimex(R)
        S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
        with pd.option_context("display.width", 200):
            print("\nmedians over sections" + (" (ratio_to_true: estimate / measured own-label bias)"
                                                 if "true_T" in R else ""))
            print(S.round(4).to_string(index=False))
            if a.condition_key and R.condition.nunique() > 1:
                cols = [c for c in ["first_level", "mcsimex_lin", "mcsimex_quad", "pairsum", "true_T"] if c in R]
                print("\nmedians by condition\n" + R.groupby("condition")[cols].median().round(4).to_string())
        print(f"wrote {a.out}/sections.csv, groups.csv and summary.csv")
    elif a.cmd == "flips":
        from .flips import flip_rates
        D = flip_rates(a.h5ad, a.label_key, a.target, a.section_key, spatial=a.spatial_key or (a.x, a.y),
                       counts_layer=a.counts_layer, sections=a.sections.split(",") if a.sections else None,
                       crop_cells=a.crop_cells, crops_per_section=a.crops_per_section, n_boot=a.n_boot,
                       seed=a.seed, labels_file=a.labels_file,
                       **({"exclude_genes": a.exclude_genes} if a.exclude_genes else {}))
        D.to_csv(a.out, index=False)
        print(f"wrote {a.out}: median flip full (published targets) {D.flip_full_pub.median():.3f}, "
              f"half {D.flip_half_rc.median():.3f}; max crop overlap {D.crop_overlap.max():.2f}")
    elif a.cmd == "simulate":
        from .simulate import make_anndata
        make_anndata(a.n_sections, seed=a.seed, label_noise=a.label_noise, n_cells=a.n_cells,
                     depth=a.depth).write_h5ad(a.out)
        print(f"wrote {a.out}: labels in obs['label'] (true labels in obs['true_label']); targets '^F'")
    elif a.cmd == "drift":
        from .drift import label_drift
        D = label_drift(a.h5ad, a.label_key, a.target, a.section_key, spatial=a.spatial_key or (a.x, a.y),
                        labeler=a.labeler,
                        counts_layer=a.counts_layer, sections=a.sections.split(",") if a.sections else None,
                        crop_cells=a.crop_cells, crops_per_section=a.crops_per_section, p=a.p, seed=a.seed,
                        labels_file=a.labels_file, **({"exclude_genes": a.exclude_genes} if a.exclude_genes else {}))
        D.to_csv(a.out, index=False)
        g = D.groupby("p")[["added_flip", "largest_share", "entropy", "dissim_added", "confusion_corr"]].median()
        print(f"wrote {a.out}; native: entropy {D.entropy_c1.median():.3f}, dissimilarity {D.dissim_native.median():.3f}")
        print(g.round(3).to_string())
    elif a.cmd == "report":
        import os
        import pandas as pd
        from .api import Result
        from .config import Config
        from .summarize import interpret
        f = os.path.join(a.out, "result.json")
        j = json.load(open(f))
        S = pd.read_csv(os.path.join(a.out, "summary.csv"))
        from .summarize import overall as _overall
        R = pd.read_csv(os.path.join(a.out, "sections.csv"))
        cfg = j["config"]
        o = _overall(R, [float(p) for p in cfg["p"]], min_depth=cfg.get("min_depth", 10.0))
        for k in ("seconds", "peak_memory_gb", "n_target_labels", "min_targets", "skipped_sections"):   # from the run
            if k in j["overall"]:
                o[k] = j["overall"][k]
        if "n_target_labels" not in o and j.get("meta", {}).get("target_labels") is not None:
            o["n_target_labels"] = len(j["meta"]["target_labels"])
        if "seconds" not in o and "seconds" in j.get("meta", {}):
            o["seconds"] = j["meta"]["seconds"]
        if a.calibrate:
            from .calibrate import calibrate as _cal
            try:
                o.update(_cal(o, None if a.calibrate == "default" else a.calibrate, a.calibration_tissue))
            except (FileNotFoundError, ValueError) as err:
                o["calibration_note"] = str(err); print(err)
        j["overall"] = o
        j["interpretation"] = interpret(o, S)          # re-derived with the installed package's wording
        json.dump(j, open(f, "w"), indent=1, default=str)
        res = Result(pd.read_csv(os.path.join(a.out, "sections.csv")), S,
                     j["overall"], j["interpretation"], Config(**j["config"]), j.get("meta", {}))
        print(res.report(os.path.join(a.out, "report.pdf")))
    elif a.cmd == "calibrate":
        from .calibrate import calibrate_dir
        try:
            c = calibrate_dir(a.out, a.table, a.tissue)
        except (FileNotFoundError, ValueError) as err:
            print(err); return 1
        if "calibration_note" in c:
            print(c["calibration_note"]); return 1
        print(f"calibrated own-label bias {c['calibrated_T']:+.4f} ({c['calibrated_share']:.0%} of the published "
              f"neighborhood share; 95% interval {c['calibrated_T_lo']:+.4f} to {c['calibrated_T_hi']:+.4f}); divisor "
              f"{c['calibration_divisor']:.2f} from {c['calibration_source']}"
              + ("; OUTSIDE the calibrated depth range" if c.get("calibration_extrapolated") else ""))
        import numpy as np
        if np.isfinite(c.get("calibrated_T_corr", np.nan)):
            print(f"truth-corrected {c['calibrated_T_corr']:+.4f} ({c['calibrated_share_corr']:.0%}; 95% interval "
                  f"{c['calibrated_T_corr_lo']:+.4f} to {c['calibrated_T_corr_hi']:+.4f}); divisor "
                  f"{c['calibration_divisor_corr']:.2f}; the two values bracket the bias")
        elif c.get("calibration_corr_note"):
            print(f"no truth correction: {c['calibration_corr_note']} (the uncorrected value tends to understate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
