"""Misclassification SIMEX for the own-label channel (experimental).

The protocol adds label error by thinning, which lets the labeler choose the confusions; at large steps the
added errors land on more dissimilar types than the native ones. Here errors are added with a chosen confusion
structure instead (misclassification SIMEX, Kuechenhoff et al. 2006), in the own-label arm only (own labels
changed, neighbor role = published labels, targets fixed by the published labels, ladder on half B):

  native confusion Pi        P(label on a Poisson resample of half A = b | re-called label c1 = a), from the same
                             resamples as the native flip rate (target cells)
  MC-SIMEX                   own labels c1 replaced by draws from Pi^lambda (lambda in --lambdas, n_sim draws
                             each); program P_T(lambda) extrapolated to lambda = -1 (linear and quadratic in lambda):
                             mcsimex_lin / mcsimex_quad = P_T(0) - P_T(-1), the own-label bias of c1
  pair sensitivity           for the most frequent native confusions a -> b: a fraction --pair-frac of the targets
                             labeled a relabeled b (slope = change in P_T per unit of added error, as a share of
                             all targets) and all of them (merge = change in P_T when a and b are merged)
  pairsum                    sum over those pairs of slope x native flip share a -> b (coverage = share of the
                             native flips the pairs cover)
  first_level                the protocol's first-level estimate at the first thinning step p0, on the same split
                             (with p0 = 0.8 and the same seed, c1, flips and first thinning equal those of a protocol
                             run whose first level is 0.8)
With known truth (truth_key; semi-synthetic or simulated data):
  true_T                     directly measured own-label bias of c1 (T1 - t, as in the protocol)
  oracle_lin / oracle_quad   MC-SIMEX with the true confusion P(c1 = b | truth = a) instead of Pi: separates the
                             extrapolation itself from the error in Pi
  pairsum_true               the same pairs, slope x true error share (truth a, c1 = b)

Per neighborhood group (groups.csv): the 12 groups of the own-label arm (k-means on the compositions of
published neighbor labels, exactly as in the ladder), with the identity each group carries,
id_g = n_g ||mean over its targets of the label mean - overall mean||^2 (size-normalized half-B expression, as a
share of the section's total sum of squares), for published, re-called and (if known) true labels:
  n, share_n           targets in the group and their share
  mix_dev              sum over labels of (group share - section share)^2, re-called labels
  flip                 native flip rate of the group's targets
  score                predicted vulnerability: n_g * sum over label pairs of ((d_a - d_b)/2)^2 ||mu_a - mu_b||^2
                       (Pi_ab + Pi_ba) * flip_g / flip_section (d: deviation of the group's label shares, mu: label
                       means; the two-type formula applied pairwise; computed before any ladder)
  resp                 identity lost when the re-called labels get one more dose of native error (Pi), measured
  loss_true            identity lost through the errors already present (true minus re-called labels; truth only)
Per section, Spearman correlations over groups: score vs resp, score vs loss_true, resp vs loss_true, and n vs
loss_true (is a group's rarity what matters?).
"""
from __future__ import annotations

import time
import zlib
from concurrent.futures import ProcessPoolExecutor

from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import numpy as np
import pandas as pd

from . import io as _io
from .config import Config
from .labeling import as_labeler
from .ladder import kmeans_best, ladder
from .protocol import section_seed

KEYS = ["true_T", "first_level", "mcsimex_lin", "mcsimex_quad", "oracle_lin", "oracle_quad", "pairsum",
        "pairsum_true"]


def matrix_power(Pi, lam):
    """Row-stochastic Pi^lam (fractional powers by scipy); negative entries clipped, rows renormalized.
    Returns (matrix, largest clipped mass in a row)."""
    if float(lam).is_integer():
        M = np.linalg.matrix_power(Pi, int(lam))
    else:
        from scipy.linalg import fractional_matrix_power
        M = np.real(fractional_matrix_power(Pi, lam))
    neg = float(np.clip(-M, 0, None).sum(1).max()) if M.size else 0.0
    M = np.clip(M, 0, None)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-12)
    return M, neg


def confusion(src, dst, idx, nlab, min_row=20):
    """Row-stochastic confusion P(dst = b | src = a) from integer codes; rows with fewer than min_row cells
    (or src codes outside 0..nlab-1) are left as identity."""
    N = np.zeros((nlab, nlab))
    ok = (src >= 0) & (dst >= 0)
    np.add.at(N, (src[ok], dst[ok]), 1.0)
    Pi = np.eye(nlab)
    rs = N.sum(1)
    good = rs >= min_row
    Pi[good] = N[good] / rs[good, None]
    return Pi, N


def draw(codes, idx, M, rng):
    """New codes for cells idx: each cell's code redrawn from row M[codes[i]]."""
    out = codes.copy()
    c = codes[idx]
    for a in np.unique(c):
        sel = idx[c == a]
        cum = np.cumsum(M[a])
        cum[-1] = 1.0
        out[sel] = np.searchsorted(cum, rng.random(len(sel)), side="right")
    return out


def _fit(lams, vals):
    lams, vals = np.asarray(lams, float), np.asarray(vals, float)
    ok = np.isfinite(vals)
    lams, vals = lams[ok], vals[ok]
    if len(lams) < 2:
        return np.nan, np.nan
    v0 = vals[lams == 0][0] if (lams == 0).any() else np.nan
    lin = np.polyval(np.polyfit(lams, vals, 1), -1.0)
    quad = np.polyval(np.polyfit(lams, vals, 2), -1.0) if len(lams) >= 3 else np.nan
    return v0 - lin, v0 - quad


def _groups(xy, auth, YB, tg, codes_own, labels, seed, cfg):
    """Neighborhood groups of the targets (as the ladder's S0 with neighbor labels = published) and the
    size-normalized half-B expression of the targets."""
    nb_codes, uniq = pd.factorize(np.asarray(auth))
    P = np.where(tg)[0]
    nbr = cKDTree(xy).query(xy, k=cfg.k + 1)[1][:, 1:]
    M = np.zeros((len(P), len(uniq)))
    nb = nb_codes[nbr[P]]
    for j in range(nb.shape[1]):
        np.add.at(M, (np.arange(len(P)), nb[:, j]), 1.0)
    M = np.sqrt(M / M.sum(1, keepdims=True))
    grp, _ = kmeans_best(M, cfg.n_groups, np.random.default_rng([seed, 999]), n_init=cfg.n_init)
    Yt = np.asarray(YB[P], dtype=np.float64)
    s = Yt.sum(1)
    ok = s > 0
    sf = s / s[ok].mean()
    y = np.where(ok[:, None], Yt / np.maximum(sf, 1e-12)[:, None], 0.0)
    return P, grp, y


def _identity_by_group(y, grp, codes, G):
    """id_g = n_g ||mean_{i in g} mu[code_i] - ybar||^2, mu = label means of y (codes: integer labels)."""
    ybar = y.mean(0)
    u, inv = np.unique(codes, return_inverse=True)
    mu = np.zeros((len(u), y.shape[1]))
    np.add.at(mu, inv, y)
    mu /= np.bincount(inv)[:, None]
    out = np.zeros(G)
    for g in range(G):
        m = grp == g
        if m.any():
            out[g] = m.sum() * ((mu[inv[m]].mean(0) - ybar) ** 2).sum()
    return out


def _group_table(sec, xy, auth, YB, tg, c1c, it, labels, fl, Pi, truth_codes, seed, cfg, rng2, n_resp=2):
    nl = len(labels)
    P, grp, y = _groups(xy, auth, YB, tg, c1c, labels, seed, cfg)
    G = cfg.n_groups
    tot = ((y - y.mean(0)) ** 2).sum()
    if tot <= 0:
        return pd.DataFrame(), {}
    fix = lambda c: np.where(c < 0, nl, c)                     # unknown labels: one extra class
    lab_ix = {l: i for i, l in enumerate(labels)}
    pub = fix(np.array([lab_ix.get(str(l), -1) for l in np.asarray(auth)[P]]))
    rc = fix(c1c[P])
    id_pub = _identity_by_group(y, grp, pub, G) / tot
    id_rc = _identity_by_group(y, grp, rc, G) / tot
    M1, _ = matrix_power(Pi, 1.0)
    resp = np.zeros(G)
    for _ in range(n_resp):
        d = fix(draw(c1c, it, M1, rng2)[P])
        resp += id_rc - _identity_by_group(y, grp, d, G) / tot
    resp /= n_resp
    # predicted score (re-called labels, native confusion; no ladder)
    n_g = np.bincount(grp, minlength=G).astype(float)
    sh = np.zeros((G, nl + 1))
    np.add.at(sh, (grp, rc), 1.0)
    sh /= np.maximum(n_g[:, None], 1)
    dev = sh - np.bincount(rc, minlength=nl + 1) / len(rc)
    u = np.unique(rc[rc < nl])
    mu = {a: y[rc == a].mean(0) for a in u}
    flip_all = float(fl[P].mean()) if len(P) else np.nan
    flip_g = np.array([fl[P][grp == g].mean() if (grp == g).any() else np.nan for g in range(G)])
    score = np.zeros(G)
    for i, a in enumerate(u):
        for b in u[i + 1:]:
            w = (Pi[a, b] + Pi[b, a]) * ((mu[a] - mu[b]) ** 2).sum() / tot
            if w > 0:
                score += n_g * ((dev[:, a] - dev[:, b]) / 2) ** 2 * w
    score *= flip_g / flip_all if flip_all > 0 else np.nan
    T = pd.DataFrame(dict(section=sec, group=np.arange(G), n=n_g, share_n=n_g / max(n_g.sum(), 1),
                          mix_dev=(dev[:, :nl] ** 2).sum(1), flip=flip_g, id_pub=id_pub, id_rc=id_rc, resp=resp,
                          score=score))
    if truth_codes is not None:
        id_true = _identity_by_group(y, grp, fix(truth_codes[P]), G) / tot
        T["id_true"], T["loss_true"] = id_true, id_true - id_rc
    rho = lambda a, b: float(spearmanr(T[a], T[b]).correlation) if T[a].nunique() > 1 and T[b].nunique() > 1 \
        else np.nan
    cor = {"grp_rho_score_resp": rho("score", "resp")}
    if truth_codes is not None:
        cor.update(grp_rho_score_loss=rho("score", "loss_true"), grp_rho_resp_loss=rho("resp", "loss_true"),
                   grp_rho_n_loss=rho("n", "loss_true"), grp_rho_mixdev_loss=rho("mix_dev", "loss_true"),
                   grp_loss_true_sum=float(T.loss_true.sum()))
    cor["grp_resp_sum"] = float(T.resp.sum())
    return T, cor


def _section_once(sec, Y, xy, auth, parent, condition, labels, is_target, labeler, cfg, opt, rep, truth):
    rng = np.random.default_rng(section_seed(cfg.seed, sec) + ([rep] if rep else []))
    seed = int(zlib.crc32(str(sec).encode()))
    Y = np.asarray(Y, dtype=np.int32)
    YA = rng.binomial(Y, 0.5).astype(np.int32)       # same draws, in the same order, as protocol._analyse_once
    YB = Y - YA
    med = float(np.median(YA.sum(1)))
    c1 = labeler.predict(YA, parent, med)
    tg = is_target(auth)
    t1 = is_target(c1)
    res = []
    fl = np.zeros(len(c1))
    for _ in range(cfg.n_boot):
        r = labeler.predict(rng.poisson(YA), parent, med)
        res.append(r)
        fl += r != c1
    flip_native = float((fl / max(cfg.n_boot, 1))[t1].mean()) if t1.any() else np.nan
    p0 = opt["p0"]
    YAt = rng.binomial(YA, p0).astype(np.int32)
    YBt = rng.binomial(YB, p0).astype(np.int32)
    cp = labeler.predict(YAt, parent, med)
    x0 = float((cp[t1] != c1[t1]).mean()) if t1.any() else np.nan

    lab_ix = {l: i for i, l in enumerate(labels)}
    code = lambda v: np.array([lab_ix.get(str(l), -1) for l in v])
    nlab = len(labels)
    c1c = code(c1)
    it = np.where(tg & (c1c >= 0))[0]                # fixed targets whose re-called label is a known label
    n_t = int(tg.sum())

    def P_T(own_labels, Yx=YB):
        return ladder(xy, own_labels, Yx, seed, tg, nb_labels=auth, k_nb=cfg.k, n_groups=cfg.n_groups,
                      n_init=cfg.n_init, clip=cfg.clip, noise=cfg.noise, n_perm=int(cfg.n_perm))["program"]

    def P_codes(codes_):
        lab = c1.copy().astype(object)
        lab[it] = labels[codes_[it]]
        return P_T(lab.astype(str))

    rec = dict(section=sec, parent=parent, condition=condition, n_cells=len(auth), n_target=n_t,
               median_counts=float(np.median(Y.sum(1))), flip_native=flip_native, x_p0=x0)
    lad = lambda own, Yx, nb: ladder(xy, own, Yx, seed, tg, nb_labels=nb, k_nb=cfg.k, n_groups=cfg.n_groups,
                                     n_init=cfg.n_init, clip=cfg.clip, noise=cfg.noise, n_perm=int(cfg.n_perm))
    A = lad(auth, YB, None)
    B = lad(auth, YBt, None)
    T1 = P_T(c1)
    Tp = P_T(cp, YBt)
    rec.update(a_program=A["program"], a_identity=A["identity"], T1_program=T1)
    L_T = (Tp - T1) - (B["program"] - A["program"])
    rec["L_T_p0"] = L_T
    rec["first_level"] = L_T / x0 * flip_native if x0 and np.isfinite(x0) and x0 > 0 else np.nan

    # native confusion among the fixed targets, from the flip resamples
    src = np.concatenate([c1c[it]] * len(res))
    dst = np.concatenate([code(r)[it] for r in res])
    Pi, Nn = confusion(src, dst, None, nlab, opt["min_row"])
    rng2 = np.random.default_rng(section_seed(cfg.seed, sec) + [rep, 77])
    lams = opt["lambdas"]

    def simex(Pm, tag):
        vals, negmax = [T1], 0.0
        for lam in lams:
            M, neg = matrix_power(Pm, lam)
            negmax = max(negmax, neg)
            vals.append(float(np.nanmean([P_codes(draw(c1c, it, M, rng2)) for _ in range(opt["n_sim"])])))
        for lam, v in zip(lams, vals[1:]):
            rec[f"{tag}_P_lam{lam:g}"] = v
        rec[f"{tag}_lin"], rec[f"{tag}_quad"] = _fit([0.0] + list(lams), vals)
        rec[f"{tag}_clipped_mass"] = negmax

    simex(Pi, "mcsimex")

    # pair sensitivity for the most frequent native confusions
    off = Nn.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
    pairs = [(a, b) for a, b in order[:opt["n_pairs"]] if off[a, b] > 0]
    n_res = max(len(res), 1)
    tot_flip = off.sum() / n_res / max(len(it), 1)
    if truth is not None:
        tc = code(truth)
        rec["acc_c1_vs_truth_targets"] = float((c1[tg] == np.asarray(truth)[tg]).mean()) if tg.any() else np.nan
    psum = psum_true = cov = 0.0
    for k, (a, b) in enumerate(pairs):
        ia = it[c1c[it] == a]
        m = int(round(opt["pair_frac"] * len(ia)))
        if m < 5:
            continue
        cc = c1c.copy()
        cc[rng2.choice(ia, m, replace=False)] = b
        slope = (P_codes(cc) - T1) / (m / max(n_t, 1))
        cc = c1c.copy()
        cc[ia] = b
        merge = P_codes(cc) - T1
        share = off[a, b] / n_res / max(len(it), 1)
        rec[f"pair{k}"] = f"{labels[a]}>{labels[b]}"
        rec[f"pair{k}_slope"], rec[f"pair{k}_merge"], rec[f"pair{k}_flipshare"] = slope, merge, share
        psum += slope * share
        cov += share
        if truth is not None:
            err = float(((tc[it] == a) & (c1c[it] == b)).sum()) / max(n_t, 1)
            rec[f"pair{k}_trueshare"] = err
            psum_true += slope * err
    rec["pairsum"] = psum
    rec["pair_coverage"] = cov / tot_flip if tot_flip > 0 else np.nan
    if truth is not None:
        rec["pairsum_true"] = psum_true
        t_arm = P_T(np.asarray(truth).astype(str))
        rec["t_program"] = t_arm
        rec["true_T"] = T1 - t_arm
        okt = tc[it] >= 0
        Pt, _ = confusion(tc[it][okt], c1c[it][okt], None, nlab, opt["min_row"])
        simex(Pt, "oracle")
    fl_frac = fl / max(cfg.n_boot, 1)
    GT, cor = _group_table(sec, xy, auth, YB, tg, c1c, it, labels, fl_frac, Pi,
                           code(truth) if truth is not None else None, seed, cfg, rng2)
    rec.update(cor)
    return rec, GT


def semisynthetic(sec, Y, parent, labeler, f, q, seed):
    """Built-in semi-synthetic split, as in the paper's semi-synthetic validation: each cell's counts are split into a
    truth share Binomial(Y, f), labeled by the labeler and treated as the true labels, and the rest, thinned to q
    and labeled again on its own counts (synthetic published labels, circular with these counts as real published
    labels are). Returns (experiment counts, synthetic published labels, true labels)."""
    rng = np.random.default_rng(section_seed(seed, sec) + [555])
    Y = np.asarray(Y, dtype=np.int32)
    YT = rng.binomial(Y, f).astype(np.int32)
    YE = Y - YT
    if q < 1:
        YE = rng.binomial(YE, q).astype(np.int32)
    truth = labeler.predict(YT, parent, float(np.median(YT.sum(1))))
    pub = labeler.predict(YE, parent, float(np.median(YE.sum(1))))
    return YE, np.asarray(pub).astype(str), np.asarray(truth).astype(str)


def _section(args):
    sec, Y, xy, auth, parent, condition, labels, is_target, labeler, cfg, opt, truth = args
    if opt.get("ss_f"):
        Y, auth, truth = semisynthetic(sec, Y, parent, labeler, opt["ss_f"], opt.get("ss_q", 1.0), cfg.seed)
    runs = [_section_once(sec, Y, xy, auth, parent, condition, labels, is_target, labeler, cfg, opt, r, truth)
            for r in range(max(1, cfg.n_splits))]
    recs = [r for r, _ in runs]
    G = pd.concat([g for _, g in runs], ignore_index=True)
    if len(G):                                        # the groups are the same in every split; values averaged
        G = G.groupby(["section", "group"], as_index=False).mean(numeric_only=True)
    if len(recs) == 1:
        return recs[0], G
    out = dict(recs[0])
    for k, v in recs[0].items():
        if k[:4] == "pair" and k[4:5].isdigit():
            continue                                  # pair details: those of the first split (pairs can differ)
        if isinstance(v, (float, np.floating)) or (isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                                                    and any(r.get(k) != v for r in recs)):
            out[k] = float(np.nanmean([r.get(k, np.nan) for r in recs]))
    out["n_splits"] = len(recs)
    return out, G


def _semisynthetic_data(data, keep, units, ref_labeler, f, q, seed):
    """The semi-synthetic data set of `units` as files would hold it: experiment counts, synthetic published labels
    (as labels) and true labels (as truth), made by the reference labeler with a fixed random stream, so
    that the split of each unit is the one `semisynthetic_train="reference"` uses."""
    import dataclasses
    import scipy.sparse as sp
    rows, C, lab, tru = [], [], [], []
    for u in units:
        ii = np.where(keep & (data.section == u))[0]
        YE, pub, truth = semisynthetic(u, data.counts[ii].toarray(), data.parent[ii][0], ref_labeler, f, q, seed)
        rows.append(ii); C.append(sp.csr_matrix(YE)); lab.append(pub); tru.append(truth)
    ii = np.concatenate(rows)
    D = dataclasses.replace(data, counts=sp.vstack(C).tocsr().astype(np.int32), label=np.concatenate(lab).astype(str),
                            section=data.section[ii], parent=data.parent[ii], condition=data.condition[ii],
                            xy=data.xy[ii], truth=np.concatenate(tru).astype(str))
    return D, np.ones(len(ii), bool)


def run_mcsimex(adata, label_key, target, section_key, spatial=("x", "y"), counts_layer="X", condition_key=None,
                sections=None, crop_cells=0, crops_per_section=2, labeler=None, truth_key=None, labels_file=None,
                exclude_genes=_io.DEFAULT_EXCLUDE, p0=0.8, lambdas=(0.5, 1.0, 1.5, 2.0), n_sim=2, n_pairs=6,
                pair_frac=0.2, min_row=20, n_jobs=1, out_csv=None, verbose=True, semisynthetic_f=None,
                semisynthetic_q=1.0, semisynthetic_train="synthetic", **settings):
    """semisynthetic_f: build the semi-synthetic split on the fly (truth share f, experiment share thinned to
    semisynthetic_q) and validate on it.

    semisynthetic_train: which labels train the labeler that re-calls labels in the protocol.
      "synthetic"  the synthetic published labels on the experiment counts, as when the protocol is run on a
                   semi-synthetic data set from files (and as on real data, where the labeler learns the published
                   labels of the counts it is given)
      "reference"  the data's own labels on the full counts, i.e. the labeler that made the truth and the synthetic
                   labels; re-called and synthetic labels then differ only by counting noise
    The truth and the synthetic published labels are made by the reference labeler in both cases."""
    from .api import _default_labeler
    from .labeling import LogRegLabeler
    cfg = Config(**settings)
    t0 = time.time()
    data = _io.load(adata, label_key, section_key, spatial=spatial, counts_layer=counts_layer,
                    condition_key=condition_key, sections=sections, crop_cells=crop_cells,
                    crops_per_section=crops_per_section, exclude_genes=exclude_genes, seed=cfg.seed,
                    labels_file=labels_file, truth_key=truth_key)
    vc = pd.Series(data.label).value_counts()
    labels = np.array(sorted(vc[vc >= cfg.min_label_cells].index))
    keep = np.isin(data.label, labels)
    is_target = _io.TargetMask(target, labels)
    if labeler == "logreg":
        lab = LogRegLabeler.from_data(data, labels, keep, seed=cfg.seed)
    else:
        lab = as_labeler(None if labeler in (None, "centroid") else labeler) or _default_labeler(data, labels, keep, cfg)
    units = [u for u in pd.unique(data.section[keep])
             if int(is_target(data.label[keep & (data.section == u)]).sum()) >= cfg.min_targets]
    opt = dict(p0=float(p0), lambdas=tuple(float(l) for l in lambdas), n_sim=int(n_sim), n_pairs=int(n_pairs),
               pair_frac=float(pair_frac), min_row=int(min_row),
               ss_f=float(semisynthetic_f) if semisynthetic_f else None, ss_q=float(semisynthetic_q))
    if semisynthetic_f and truth_key:
        raise ValueError("semisynthetic_f and truth_key are alternatives")
    if semisynthetic_train not in ("synthetic", "reference"):
        raise ValueError("semisynthetic_train: 'synthetic' or 'reference'")
    if semisynthetic_f and semisynthetic_train == "synthetic":
        data, keep = _semisynthetic_data(data, keep, units, lab, opt["ss_f"], opt["ss_q"], cfg.seed)
        if labeler == "logreg":
            lab = LogRegLabeler.from_data(data, labels, keep, seed=cfg.seed)
        else:
            lab = as_labeler(None if labeler in (None, "centroid") else labeler) or \
                _default_labeler(data, labels, keep, cfg)
        opt["ss_f"] = None                                # the split is in `data` now; truth in data.truth

    def job(u):
        ii = np.where(keep & (data.section == u))[0]
        return (u, data.counts[ii].toarray(), data.xy[ii], data.label[ii], data.parent[ii][0], data.condition[ii][0],
                labels, is_target, lab, cfg, opt, data.truth[ii] if data.truth is not None else None)

    recs, groups = [], []
    if n_jobs > 1 and len(units) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            futs = {ex.submit(_section, job(u)): u for u in units}
            for f in futs:
                r, g = f.result()
                recs.append(r); groups.append(g)
                if verbose:
                    _progress(recs[-1], t0)
    else:
        for u in units:
            r, g = _section(job(u))
            recs.append(r); groups.append(g)
            if verbose:
                _progress(recs[-1], t0)
    R = pd.DataFrame(recs)
    G = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    R.attrs["groups"] = G
    if out_csv:
        R.to_csv(out_csv, index=False)
        G.to_csv(out_csv.replace("sections.csv", "groups.csv") if out_csv.endswith("sections.csv")
                 else out_csv + ".groups.csv", index=False)
    return R


def _progress(r, t0):
    s = " | ".join(f"{k} {r[k]:+.4f}" for k in KEYS if k in r and np.isfinite(r[k]))
    print(f"  {r['section']}: flip {r['flip_native']:.3f} | {s} [{time.time() - t0:.0f} s]", flush=True)


def summarize_mcsimex(R):
    """Medians over sections, and each estimate over the measured bias where the truth is known."""
    rows = []
    for k in KEYS + ["a_program", "flip_native", "pair_coverage", "grp_rho_score_resp", "grp_rho_score_loss",
                     "grp_rho_resp_loss", "grp_rho_n_loss", "grp_rho_mixdev_loss", "grp_resp_sum",
                     "grp_loss_true_sum"]:
        if k in R:
            rows.append(dict(quantity=k, median=float(R[k].median()), n_pos=int((R[k] > 0).sum()),
                             n=int(R[k].notna().sum())))
    S = pd.DataFrame(rows)
    if "true_T" in R:
        for k in KEYS[1:]:
            if k in R:
                S.loc[S.quantity == k, "ratio_to_true"] = float(R[k].median() / R["true_T"].median())
    return S
