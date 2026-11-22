"""One section through the protocol: count splitting, re-called labels, thinning, and the ladder (and
optionally the NCEM-style model) scored on every arm.

Arms (each: own labels, counts, neighbor-role labels, target rule)
  a         published labels                       c1        re-called labels
  b<p>      published labels, thinned              c<p>      labels re-called on thinned counts
  c1self, c<p>self   re-called labels scored on the half they were called from (circularity)
  T1, T<p>  own = re-called, neighbor role = published   (own-label channel)
  N1, N<p>  own = published, neighbor role = re-called   (neighbor channel)
  c1R, c<p>R re-called labels with the target set re-selected by them   (membership channel)
Label effect of any quantity q (program, identity, S0, ...):
  L_q_p      = (c<p> - c1) - (b<p> - a)
  L_T_q_p    = (T<p> - T1) - (b<p> - a),  L_N_q_p likewise,  L_TN_q_p = L - L_T - L_N
  L_M_q_p    = [(c<p>R - c1R) - (b<p> - a)] - L_q_p          (re-selected minus fixed targets)
  circ_q_native = c1self - c1,  circ_q_p = c<p>self - c<p>
Validation arms, only when true labels are known (simulated or semi-synthetic data; truth_key):
  t         own = true, neighbor role = published          tt   true labels in both roles
  tc        own = re-called, neighbor role = true
  directly measured bias of the re-called labels c1 (same half B, same split, fixed targets):
  true_net_q = c1 - tt,   true_T_q = T1 - t,   true_N_q = c1 - tc,   true_TN_q = net - T - N
  and of the published labels (scored on half B, so partly circular): true_pub_net_q = a - tt
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.spatial import cKDTree

from .ladder import ladder
from .ncem import ncem as ncem_score

LADDER_Q = ["program", "identity", "S0", "S3"]
NCEM_Q = ["ncem_niche", "ncem_type", "ncem_full"]


def section_seed(seed, section):
    return [int(seed), zlib.crc32(str(section).encode())]


SD_KEYS = ("L_program_", "L_T_program_", "L_N_program_", "L_M_program_", "L_ncem_niche_", "circ_identity_native",
           "circ_program_native", "true_net_program", "true_T_program", "true_N_program")


def analyse_section(sec, Y, xy, auth, parent, condition, labels, is_target, labeler, cfg, truth=None):
    """Returns (record, context) for one section, averaged over cfg.n_splits independent random replicates
    (count split, thinning, resampling). Split 0 uses the same random stream as a single-split run. For the
    main label effects the standard deviation across splits is kept as sd_<key>."""
    K = max(1, int(getattr(cfg, "n_splits", 1)))
    runs = [_analyse_once(sec, Y, xy, auth, parent, condition, labels, is_target, labeler, cfg, rep=r, truth=truth)
            for r in range(K)]
    if K == 1:
        return runs[0]
    recs = [r for r, _ in runs]
    rec = dict(recs[0])
    for k, v in recs[0].items():
        if isinstance(v, (int, np.integer)) and all(r.get(k) == v for r in recs):
            continue                                  # counts identical in every split (n_cells, n_target)
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            vals = np.array([r.get(k, np.nan) for r in recs], float)
            rec[k] = float(np.nanmean(vals))
            if k.startswith(SD_KEYS):
                rec[f"sd_{k}"] = float(np.nanstd(vals, ddof=1))
    rec["n_splits"] = K
    return rec, runs[0][1]


def _analyse_once(sec, Y, xy, auth, parent, condition, labels, is_target, labeler, cfg, rep=0, truth=None):
    """One random replicate of the protocol for one section."""
    rng = np.random.default_rng(section_seed(cfg.seed, sec) + ([rep] if rep else []))
    seed = int(zlib.crc32(str(sec).encode()))
    Y = np.asarray(Y, dtype=np.int32)                 # 32-bit counts: half the memory of int64, same values
    if cfg.split:
        YA = rng.binomial(Y, 0.5).astype(np.int32)
        YB = Y - YA
    else:
        YA = YB = Y
    med = float(np.median(YA.sum(1)))
    c1 = labeler.predict(YA, parent, med)
    tg_pub = is_target(auth)
    ncem_g = (np.argsort(-Y[tg_pub].mean(0))[:cfg.ncem_genes] if cfg.ncem and cfg.ncem_genes and tg_pub.any()
              else None)
    rec = dict(section=sec, parent=parent, condition=condition, n_cells=len(auth), n_target=int(tg_pub.sum()),
               median_counts=float(np.median(Y.sum(1))), median_counts_ladder=float(np.median(YB.sum(1))),
               acc_c1_vs_published=float((c1 == auth).mean()),
               acc_c1_targets=float((c1[tg_pub] == auth[tg_pub]).mean()) if tg_pub.any() else np.nan)
    t1 = is_target(c1)
    fl = np.zeros(len(c1))
    for _ in range(cfg.n_boot):
        fl += labeler.predict(rng.poisson(YA), parent, med) != c1
    rec["flip_native"] = float((fl / max(cfg.n_boot, 1))[t1].mean()) if t1.any() else np.nan

    arms = {"a": (auth, YB, None, False), "c1": (c1, YB, None, False)}
    if cfg.split:
        arms["c1self"] = (c1, YA, None, False)
    if cfg.channels:
        arms["T1"] = (c1, YB, auth, False)
        arms["N1"] = (auth, YB, c1, False)
    if cfg.membership and cfg.fixed_targets:
        arms["c1R"] = (c1, YB, None, True)
    if truth is not None:                             # validation arms (known truth); no random draws, so the
        truth = np.asarray(truth).astype(str)         # other arms are unchanged by adding them
        arms["tt"] = (truth, YB, None, False)
        arms["t"] = (truth, YB, auth, False)
        arms["tc"] = (c1, YB, truth, False)
        rec["acc_c1_vs_truth_targets"] = float((c1[tg_pub] == truth[tg_pub]).mean()) if tg_pub.any() else np.nan
        rec["acc_published_vs_truth_targets"] = (float((auth[tg_pub] == truth[tg_pub]).mean()) if tg_pub.any()
                                                 else np.nan)
    cps = {}
    for p in cfg.p:
        YAt = rng.binomial(YA, p).astype(np.int32)
        YBt = rng.binomial(YB, p).astype(np.int32) if cfg.split else YAt
        cp = labeler.predict(YAt, parent, med)
        cps[p] = cp
        rec[f"x_{p}"] = float((cp[t1] != c1[t1]).mean()) if t1.any() else np.nan
        arms[f"b{p}"] = (auth, YBt, None, False)
        arms[f"c{p}"] = (cp, YBt, None, False)
        if cfg.split:
            arms[f"c{p}self"] = (cp, YAt, None, False)
        if cfg.channels:
            arms[f"T{p}"] = (cp, YBt, auth, False)
            arms[f"N{p}"] = (auth, YBt, cp, False)
        if cfg.membership and cfg.fixed_targets:
            arms[f"c{p}R"] = (cp, YBt, None, True)

    for arm, (own, Yx, nbl, reselect) in arms.items():
        tmask = is_target(own) if (reselect or not cfg.fixed_targets) else tg_pub
        for k, v in ladder(xy, own, Yx, seed, tmask, nb_labels=nbl, k_nb=cfg.k, n_groups=cfg.n_groups,
                           n_init=cfg.n_init, clip=cfg.clip, noise=cfg.noise,
                           n_perm=int(getattr(cfg, "n_perm", 1))).items():
            rec[f"{arm}_{k}"] = v
        if cfg.ncem:
            for k, v in ncem_score(xy, own, Yx, tmask, labels, seed, nb_labels=nbl, k_nb=cfg.k,
                                   genes=ncem_g).items():
                rec[f"{arm}_{k}"] = v

    Q = LADDER_Q + (NCEM_Q if cfg.ncem else [])
    g = lambda arm, q: rec.get(f"{arm}_{q}", np.nan)
    for p in cfg.p:
        for q in Q:
            depth_ = g(f"b{p}", q) - g("a", q)
            L = (g(f"c{p}", q) - g("c1", q)) - depth_
            rec[f"L_{q}_{p}"] = L
            if cfg.channels:
                rec[f"L_T_{q}_{p}"] = (g(f"T{p}", q) - g("T1", q)) - depth_
                rec[f"L_N_{q}_{p}"] = (g(f"N{p}", q) - g("N1", q)) - depth_
                rec[f"L_TN_{q}_{p}"] = L - rec[f"L_T_{q}_{p}"] - rec[f"L_N_{q}_{p}"]
            if cfg.membership and cfg.fixed_targets:
                rec[f"L_M_{q}_{p}"] = ((g(f"c{p}R", q) - g("c1R", q)) - depth_) - L
    if truth is not None:
        for q in Q:
            rec[f"true_net_{q}"] = g("c1", q) - g("tt", q)
            rec[f"true_T_{q}"] = g("T1", q) - g("t", q) if cfg.channels else np.nan
            rec[f"true_N_{q}"] = g("c1", q) - g("tc", q)
            rec[f"true_TN_{q}"] = rec[f"true_net_{q}"] - rec[f"true_T_{q}"] - rec[f"true_N_{q}"]
            rec[f"true_pub_net_{q}"] = g("a", q) - g("tt", q)
    if cfg.split:
        for q in Q:
            rec[f"circ_{q}_native"] = g("c1self", q) - g("c1", q)
            for p in cfg.p:
                rec[f"circ_{q}_{p}"] = g(f"c{p}self", q) - g(f"c{p}", q)

    ctx = None
    if cfg.context_tests and t1.any():
        it = np.where(t1)[0]
        nb = cKDTree(xy).query(xy[it], k=cfg.k + 1)[1][:, 1:]
        codes = {l: i for i, l in enumerate(labels)}
        ca = np.array([codes.get(l, -1) for l in auth])
        M = np.zeros((len(it), len(labels)))
        for j in range(cfg.k):
            cj = ca[nb[:, j]]
            ok = cj >= 0
            np.add.at(M, (np.arange(len(it))[ok], cj[ok]), 1.0)
        ctx = dict(section=sec, parent=parent, F=np.sqrt(M / cfg.k), c1=c1[it], auth=auth[it],
                   depth=np.log1p(Y[it].sum(1)), cp={p: cps[p][it] for p in cfg.p})
        # neighborhood profiles of EVERY label from all cells of the section (sums and counts, pooled over
        # sections in context_tests), so the destination test also covers changes to non-target labels and
        # single-label target sets. The original test, with profiles from the targets only, is unchanged.
        nb_all = cKDTree(xy).query(xy, k=cfg.k + 1)[1][:, 1:]
        M_all = np.zeros((len(xy), len(labels)))
        for j in range(cfg.k):
            cj = ca[nb_all[:, j]]
            ok = cj >= 0
            np.add.at(M_all, (np.arange(len(xy))[ok], cj[ok]), 1.0)
        F_all = np.sqrt(M_all / cfg.k)
        c1c = np.array([codes.get(l, -1) for l in c1])
        okc = c1c >= 0
        psum = np.zeros((len(labels), len(labels)))
        np.add.at(psum, c1c[okc], F_all[okc])
        ctx.update(labels=np.asarray(labels), prof_sum=psum, prof_n=np.bincount(c1c[okc], minlength=len(labels)))
    return rec, ctx
