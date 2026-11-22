"""Pooled summaries: per thinning level medians, context tests, slopes and the first-level estimate, plus a short
plain-language interpretation of what the diagnostic can and cannot say."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum()) if ok.any() and (x[ok] ** 2).sum() > 0 else np.nan


def consistent_sign(k_pos, n, share=0.75):
    """+1 if at least `share` of n units are positive, -1 if at least `share` are not, else 0. Below four
    units no direction is called."""
    if not n or n < 4:
        return 0
    return 1 if k_pos >= share * n else (-1 if (n - k_pos) >= share * n else 0)


def context_tests(ctx_rows, ps, seed=0, n_null=50):
    """Occurrence: does neighbor composition predict WHICH target cells change label (section-held-out AUC
    gain over own label + depth)? Destination: do changes land on the label typical of the neighborhood
    more often than the same transitions given to random cells with the same label? Restricted to cells
    whose re-called label agrees with the published one (destination test only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ctx_rows = [c for c in ctx_rows if c is not None and len(c["c1"])]
    if not ctx_rows:
        return {}
    rng = np.random.default_rng(seed)
    F = np.vstack([c["F"] for c in ctx_rows])
    c1 = np.concatenate([c["c1"] for c in ctx_rows])
    agree = c1 == np.concatenate([c["auth"] for c in ctx_rows])
    dep = np.concatenate([c["depth"] for c in ctx_rows])
    grp = np.concatenate([[c["parent"]] * len(c["c1"]) for c in ctx_rows])
    own = pd.get_dummies(pd.Series(c1)).to_numpy(float)
    prof = pd.DataFrame(F).groupby(c1).mean()
    out = {}
    for p in ps:
        new = np.concatenate([c["cp"][p] for c in ctx_rows])
        y = (new != c1).astype(int)

        def cv(X):
            if len(np.unique(grp)) < 2 or y.min() == y.max():
                return np.nan
            pred = np.zeros(len(y))
            for g in np.unique(grp):
                te = grp == g
                if y[~te].min() == y[~te].max():
                    return np.nan
                pred[te] = LogisticRegression(max_iter=2000).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
            return roc_auc_score(y, pred)

        auc_b, auc_f = cv(np.column_stack([own, dep])), cv(np.column_stack([own, dep, F]))
        fl = np.where((y == 1) & np.isin(new, prof.index) & agree)[0]
        rec = dict(auc_base=auc_b, auc_full=auc_f, auc_gain=auc_f - auc_b, n_flips_agree=len(fl))
        if len(fl):
            d_old = np.linalg.norm(F[fl] - prof.loc[c1[fl]].to_numpy(), axis=1)
            d_new = np.linalg.norm(F[fl] - prof.loc[new[fl]].to_numpy(), axis=1)
            pools = {l: np.where((c1 == l) & agree)[0] for l in np.unique(c1[fl])}
            null = []
            for _ in range(n_null):
                hit = []
                for l, pool in pools.items():
                    m = c1[fl] == l
                    cells = rng.choice(pool, m.sum(), replace=True)
                    hit.append(np.linalg.norm(F[cells] - prof.loc[new[fl][m]].to_numpy(), axis=1)
                               < np.linalg.norm(F[cells] - prof.loc[l].to_numpy(), axis=1))
                null.append(np.concatenate(hit).mean())
            rec.update(toward_context=float((d_new < d_old).mean()), toward_null=float(np.mean(null)),
                       toward_null_lo=float(np.percentile(null, 2.5)), toward_null_hi=float(np.percentile(null, 97.5)))
        out[p] = rec
    _destination_all_labels(ctx_rows, ps, out, F, c1, agree, seed, n_null)
    return out


def _destination_all_labels(ctx_rows, ps, out, F, c1, agree, seed, n_null):
    """The destination test with neighborhood profiles of every label, built from all cells (not only
    the targets). It covers label changes that leave the target set and target sets of a single label, where
    the original test is empty. Written as toward_context_all, toward_null_all(_lo, _hi), n_flips_all; uses its
    own random stream, so the original columns are unchanged."""
    if not all("prof_sum" in c for c in ctx_rows):
        return                                         # sections cached without the all-label profiles
    labs = ctx_rows[0]["labels"]
    if any(len(c["labels"]) != len(labs) or np.any(c["labels"] != labs) for c in ctx_rows):
        return
    ps_ = np.sum([c["prof_sum"] for c in ctx_rows], axis=0)
    pn = np.sum([c["prof_n"] for c in ctx_rows], axis=0)
    have = pn > 0
    prof = pd.DataFrame(ps_[have] / pn[have, None], index=labs[have])
    rng = np.random.default_rng([int(seed), 28])
    for p in ps:
        new = np.concatenate([c["cp"][p] for c in ctx_rows])
        fl = np.where((new != c1) & np.isin(new, prof.index) & np.isin(c1, prof.index) & agree)[0]
        rec = out.setdefault(p, {})
        rec["n_flips_all"] = len(fl)
        if not len(fl):
            continue
        d_old = np.linalg.norm(F[fl] - prof.loc[c1[fl]].to_numpy(), axis=1)
        d_new = np.linalg.norm(F[fl] - prof.loc[new[fl]].to_numpy(), axis=1)
        pools = {l: np.where((c1 == l) & agree)[0] for l in np.unique(c1[fl])}
        null = []
        for _ in range(n_null):
            hit = []
            for l, pool in pools.items():
                m = c1[fl] == l
                cells = rng.choice(pool, m.sum(), replace=True)
                hit.append(np.linalg.norm(F[cells] - prof.loc[new[fl][m]].to_numpy(), axis=1)
                           < np.linalg.norm(F[cells] - prof.loc[l].to_numpy(), axis=1))
            null.append(np.concatenate(hit).mean())
        rec.update(toward_context_all=float((d_new < d_old).mean()), toward_null_all=float(np.mean(null)),
                   toward_null_all_lo=float(np.percentile(null, 2.5)),
                   toward_null_all_hi=float(np.percentile(null, 97.5)))


def summarize(R, ps, ctx=None):
    """Per thinning level: medians over sections of every label effect and circularity term, counts of
    sections by sign, and pooled slopes. Returns a DataFrame with one row per p."""
    rows = []
    for p in ps:
        rec = dict(p=p, effective_depth=float(R.median_counts_ladder.median() * p),
                   x_median=float(R[f"x_{p}"].median()))
        for col in [c for c in R.columns if c.endswith(f"_{p}") and (c.startswith(("L_", "circ_")))]:
            rec[col[: -len(f"_{p}")]] = float(R[col].median())
        rec["n_net_pos"] = int((R[f"L_program_{p}"] > 0).sum())
        if f"L_T_program_{p}" in R:
            rec["n_T_pos"] = int((R[f"L_T_program_{p}"] > 0).sum())
            rec["n_N_neg"] = int((R[f"L_N_program_{p}"] < 0).sum())
        rec["slope"] = _slope(R[f"x_{p}"], R[f"L_program_{p}"])
        if ctx and p in ctx:
            rec.update(ctx[p])
        rows.append(rec)
    return pd.DataFrame(rows)


def usable_levels(R, ps, min_depth=10.0):
    """Thinning levels whose effective depth (median counts per cell the ladder sees x p) is >= min_depth."""
    d = float(R.median_counts_ladder.median())
    return [p for p in ps if d * p >= min_depth]


def overall(R, ps, min_depth=10.0):
    """Dataset-level numbers: native quantities, pooled slopes, the first-level estimate, checks. The keys
    lower_bound_* (slope x native flip) keep their historical name for compatibility; validation on semi-synthetic data
    showed they are not bounds, and the report uses the first-level estimate instead. Slopes
    use only levels with effective depth >= min_depth (channel signs are unreliable below ~10 counts per cell
    in simulations); the all-level versions are kept as *_all_levels for comparison."""
    o = dict(n_sections=len(R), median_counts=float(R.median_counts.median()),
             ladder_depth=float(R.median_counts_ladder.median()), flip_native=float(R.flip_native.median()),
             acc_c1_vs_published=float(R.acc_c1_vs_published.median()),
             identity_published=float(R.a_identity.median()), program_published=float(R.a_program.median()))
    use = usable_levels(R, ps, min_depth)
    o["min_depth"] = float(min_depth)
    o["n_levels_used"] = len(use)
    o["levels_below_min_depth"] = ",".join(f"{p:g}" for p in ps if p not in use)
    for tag, col in [("net", "L_program"), ("T", "L_T_program")]:
        if f"{col}_{ps[0]}" in R:
            for suf, lv in [("", use), ("_all_levels", list(ps))]:
                if not lv:
                    o[f"slope_{tag}{suf}"] = o[f"lower_bound_{tag}{suf}"] = np.nan
                    continue
                sl = _slope(np.concatenate([R[f"x_{p}"] for p in lv]),
                            np.concatenate([R[f"{col}_{p}"] for p in lv]))
                o[f"slope_{tag}{suf}"] = sl
                o[f"lower_bound_{tag}{suf}"] = sl * o["flip_native"]
    # first-level estimate: own-label (and net) response per added flip at the first thinning level x native flip
    p0 = ps[0]
    if f"L_T_program_{p0}" in R and f"x_{p0}" in R:
        for tag, col in [("T", "L_T_program"), ("net", "L_program")]:
            e = R[f"{col}_{p0}"] / R[f"x_{p0}"] * R.flip_native
            o[f"first_level_{tag}"] = float(e.median())
        o["first_level_T_n_pos"] = int(((R[f"L_T_program_{p0}"] / R[f"x_{p0}"]) > 0).sum())
        o["first_level_p"] = float(p0)
        o["first_level_depth"] = o["ladder_depth"] * float(p0)
    for rung in ["S0", "S1"]:                           # share of genes clipped (published labels, arm a)
        if f"a_negfrac_{rung}" in R:
            o[f"clip_frac_{rung}"] = float(R[f"a_negfrac_{rung}"].median())
    if "circ_identity_native" in R:
        for q in ["identity", "program", "S0", "ncem_niche"]:
            if f"circ_{q}_native" in R:
                o[f"circ_{q}_native"] = float(R[f"circ_{q}_native"].median())
        # circ_ncem_type / circ_ncem_full are not reported: with fitted labels the noise-corrected type
        # effect is undefined (it can exceed 1), which is itself the demonstration of circularity
    if "true_net_program" in R:                        # validation runs with known true labels
        for c in [c for c in R.columns if c.startswith(("true_", "acc_c1_vs_truth", "acc_published_vs_truth"))
                  and not c.startswith("sd_")]:
            o[c] = float(R[c].median())
    if f"L_T_S0_{ps[0]}" in R:
        o["check_max_abs_L_T_S0"] = float(max(np.nanmax(np.abs(R[f"L_T_S0_{p}"])) for p in ps))
    return o


def interpret(o, S, low_identity=0.02):
    """Plain-language reading, with the limits established in the accompanying paper."""
    lines = []
    mid = S.iloc[len(S) // 2]
    md = o.get("min_depth", 10.0)
    low = [f"{r.p:g}" for _, r in S.iterrows() if r.effective_depth < md]
    if low:
        lines.append(f"WARNING: thinning level(s) {', '.join(low)} leave fewer than {md:g} counts per cell for the "
                     "model; channel signs there are unreliable (simulations) and they are excluded from the "
                     "pooled slopes.")
    n_tl = o.get("n_target_labels")                   # set by diagnose (None if absent)
    single = n_tl == 1
    low_I = o["identity_published"] < low_identity
    n = o["n_sections"]
    if "L_T_program" in S:
        t = mid["L_T_program"]
        k = int(mid.n_T_pos)
        direction = consistent_sign(k, n)                 # +1, -1 or 0 (no consistent direction)
        if single:
            why = ("With a single target label the published labels carry no identity among the targets, so the "
                   "own-label channel cannot measure a bias of these labels; its value reflects noise and the "
                   "re-called labels of other types. Define the targets as two or more labels (subtypes or states) "
                   "to measure it.")
        elif direction > 0 and t > 0:
            why = ("Added label error in the targets' own labels increased the estimated neighborhood share "
                   "(a sensitivity to label perturbation, not the native bias itself).")
        elif direction == 0:
            why = (f"Added label error in the targets' own labels changed the estimated neighborhood share in no "
                   f"consistent direction ({k} of {n} sections > 0)"
                   + (f"; with {n} sections a direction cannot be established." if n < 4 else
                      (f", as expected when target identity is near zero (here {o['identity_published']:.3f})."
                       if low_I else "; treat the sign as undetermined.")))
        elif low_I:
            why = ("Added label error in the targets' own labels did not increase the estimated neighborhood share, "
                   f"as expected when target identity is near zero (here {o['identity_published']:.3f}).")
        else:
            why = ("Added label error in the targets' own labels did not increase the estimated neighborhood share "
                   f"although target identity is not small ({o['identity_published']:.3f}); this is unexpected - "
                   "check the depth at this level and the agreement of re-called with published labels.")
        lines.append(f"Own-label channel at {mid.p:g}: {t:+.4f} ({k}/{n} sections > 0). " + why)
        lines.append(f"Net label effect at {mid.p:g}: {mid.L_program:+.4f}; neighbor channel {mid.L_N_program:+.4f}"
                     + (f"; membership channel {mid.L_M_program:+.4f}" if "L_M_program" in S else "") + ".")
    if np.isfinite(o.get("first_level_T", np.nan)) and single:
        lines.append("First-level estimate of the own-label bias: not defined for a single target label.")
    elif np.isfinite(o.get("first_level_T", np.nan)) and low_I:
        lines.append(f"First-level estimate of the own-label bias: {o['first_level_T']:+.4f} "
                     f"({o['first_level_T_n_pos']}/{n} sections > 0), but target identity is low "
                     f"({o['identity_published']:.3f} < {low_identity:g}): the own-label channel has little to act "
                     "on, and neither its sign nor its size is informative here.")
    elif np.isfinite(o.get("first_level_T", np.nan)):
        txt = (f"First-level estimate of the own-label bias in the neighborhood share (response per added flip at the first "
               f"thinning level p = {o['first_level_p']:g}, times the native flip rate): {o['first_level_T']:+.4f} "
               f"vs published neighborhood share {o['program_published']:.3f} "
               f"({o['first_level_T_n_pos']}/{n} sections > 0). Its sign was reliable on semi-synthetic "
               "data built from real tissue; its magnitude is first-order: the native flip rate understates the error rate "
               "(about 0.75-fold in validation), and a large first step overstates the response in deep data.")
        if consistent_sign(o["first_level_T_n_pos"], n) == 0:
            txt += " The sign is not consistent over sections here, so treat it as undetermined."
        if o["first_level_p"] < 0.75:
            txt += (" For a magnitude, rerun with a first step of p = 0.8 (e.g. --p 0.8,0.65,0.5) and use "
                    "'nichepeek calibrate'.")
        lines.append(txt)
    if np.isfinite(o.get("calibrated_T", np.nan)):
        lines.append(f"Calibrated own-label bias, uncorrected: {o['calibrated_T']:+.4f} ({o['calibrated_share']:.0%} of the "
                     f"published neighborhood share; 95% interval {o['calibrated_T_lo']:+.4f} to {o['calibrated_T_hi']:+.4f}), "
                     f"dividing by the estimate/measured ratio {o['calibration_divisor']:.2f} of "
                     f"{o['calibration_source']} (ladder depth {o.get('calibration_depth', float('nan')):.0f} counts "
                     f"per cell); with any other calibration row of the same first step, "
                     f"{o.get('calibrated_T_rows_lo', float('nan')):+.4f} to {o.get('calibrated_T_rows_hi', float('nan')):+.4f}."
                     + (f" This dataset (ladder depth {o['ladder_depth']:.0f}) lies outside the depth range of the "
                        "calibration, so the value is an extrapolation." if o.get("calibration_extrapolated") else ""))
        if np.isfinite(o.get("calibrated_T_corr", np.nan)):
            lines.append(f"Corrected for errors in the truth labels of the semi-synthetic data (divisor "
                         f"{o['calibration_divisor_corr']:.2f}): {o['calibrated_T_corr']:+.4f} "
                         f"({o['calibrated_share_corr']:.0%} of the published neighborhood share; 95% interval "
                         f"{o['calibrated_T_corr_lo']:+.4f} to {o['calibrated_T_corr_hi']:+.4f}, units only). The "
                         f"uncorrected and corrected values bracket the own-label bias: {o['calibrated_share']:.0%} to "
                         f"{o['calibrated_share_corr']:.0%} of the neighborhood share.")
        elif o.get("calibration_corr_note"):
            lines.append(f"No truth correction ({o['calibration_corr_note']}). Errors in the truth labels shrink the "
                         "measured bias, so the uncorrected value tends to understate the own-label bias.")
    if np.isfinite(o.get("clip_frac_S0", np.nan)):
        lines.append(f"Clipping: {o['clip_frac_S0']:.0%} of genes at S0 and {o.get('clip_frac_S1', np.nan):.0%} at "
                     "S1 had a negative residual variance, set to zero; --no-clip gives the unclipped estimate.")
    if "circ_identity_native" in o:
        lines.append(f"Circularity (labels fitted to the counts they are scored on): identity "
                     f"{o['circ_identity_native']:+.4f}, neighborhood share {o['circ_program_native']:+.4f} at "
                     f"{o['ladder_depth']:.0f} counts per cell. Any pipeline that labels and models the same counts "
                     "carries this bias; count splitting removes it when technical noise is Poisson-like, but not "
                     "segmentation spillover, which is shared by both halves.")
    if "auc_gain" in S and np.isfinite(mid.get("auc_gain", np.nan)):
        lines.append(f"Occurrence: label changes are {'barely ' if mid.auc_gain < 0.02 else ''}predicted by the "
                     f"neighborhood (section-held-out AUC gain {mid.auc_gain:+.3f}).")
    if "toward_context" in S and np.isfinite(mid.get("toward_context", np.nan)):
        lines.append(f"Destination: label changes land on context-typical labels {mid.toward_context:.2f} vs null "
                     f"{mid.toward_null:.2f} (consistent with context-dependent expression, and also with "
                     "segmentation spillover from neighbors).")
    elif "toward_context_all" in S and np.isfinite(mid.get("toward_context_all", np.nan)):
        lines.append(f"Destination (profiles of all labels): label changes land on context-typical labels "
                     f"{mid.toward_context_all:.2f} vs null {mid.toward_null_all:.2f} (consistent with "
                     "context-dependent expression, and also with segmentation spillover from neighbors).")
    elif "n_flips_agree" in S and mid.get("n_flips_agree", 1) == 0:
        lines.append("Destination: no label change among the targets could be tested (the test needs label "
                     "changes between labels with a neighborhood profile).")
    if o.get("skipped_sections"):
        lines.append(f"Skipped {len(o['skipped_sections'])} section(s) with fewer than {o.get('min_targets', 50)} "
                     f"target cells: {', '.join(map(str, o['skipped_sections']))}.")
    if o.get("check_max_abs_L_T_S0", 0) > 1e-9:
        lines.append("WARNING: the own-label channel changed S0; this should be impossible - please report.")
    return lines
