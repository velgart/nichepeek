"""The permutation ladder (S0 / S1 / S3) with the two roles of a cell-type label separated.

The third rung (cells relocated at random) is S_2 in the accompanying paper; output columns use the name S3
(e.g. a_S3).

A label enters the ladder twice: as the cell's OWN identity (the classes within which S1 permutes
expression) and as a NEIGHBOR label (the composition that defines the neighborhood groups at S0).
`ladder` takes both explicitly, plus the mask of queried (target) cells, so that label error can be
routed through one channel at a time.

    identity = R2(S1) - R2(S3)        program = R2(S0) - R2(S1)

Ported from scramble_ladder3.py / sim_tissue.ladder / diagnose_dataset.ladder_ch; with nb_labels=None
and tgt = is_target(own) the output is identical to the original ladder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NOISE_MODELS = ("poisson", "multinomial")


def permute_within(codes, rng):
    """Permutation of positions that only exchanges positions with identical codes."""
    n = len(codes)
    perm = np.arange(n)
    order = np.argsort(codes, kind="stable")
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    for idx in np.split(order, cuts):
        if len(idx) > 1:
            perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def kmeans(M, G, rng, iters=12, init=None):
    cent = init.copy() if init is not None else M[rng.choice(len(M), min(G, len(M)), replace=False)].copy()
    lab = np.zeros(len(M), int)
    for _ in range(iters):
        lab = np.argmin(((M[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        for g in range(len(cent)):
            if (lab == g).any():
                cent[g] = M[lab == g].mean(axis=0)
    return lab, cent


def kmeans_pp(M, G, rng):
    cent = [M[rng.integers(len(M))]]
    d2 = ((M - cent[0]) ** 2).sum(axis=1)
    for _ in range(1, min(G, len(M))):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(M[rng.choice(len(M), p=p)])
        d2 = np.minimum(d2, ((M - cent[-1]) ** 2).sum(axis=1))
    return np.array(cent)


def kmeans_best(M, G, rng, n_init=10, iters=30):
    """Best-inertia partition over n_init k-means++ restarts."""
    best = None
    for _ in range(n_init):
        lab, cent = kmeans(M, G, None, iters=iters, init=kmeans_pp(M, G, rng))
        inertia = ((M - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, cent)
    return best[1], best[2]


def sampling_variance(Y, sf, model="poisson"):
    """Per-gene sampling variance of size-normalized counts Y / sf.

    poisson      mean(Y / sf^2)                      (as in the original ladder)
    multinomial  mean(Y (t - Y) / ((t - 1) sf^2))    unbiased given the cell total t, which the size
                 factor conditions on; the Poisson form overstates it by 1 / (1 - p_g)."""
    if model == "poisson":
        return (Y / (sf ** 2)[:, None]).mean(axis=0)
    if model == "multinomial":
        t = np.maximum(Y.sum(1), 2)[:, None]
        return (Y * (t - Y) / (t - 1) / (sf ** 2)[:, None]).mean(axis=0)
    raise ValueError(f"noise model must be one of {NOISE_MODELS}")


def r2_micro(Y, s, lab, noise="poisson"):
    """Micro-averaged R2 of the biological variance explained by the grouping `lab`.
    Returns (clipped R2, unclipped R2, fraction of genes with negative residual, n genes).
    s: per-cell total counts (size factors are s / mean(s))."""
    good = s > 0
    Y, s, lab = Y[good].astype(np.float64), s[good], lab[good]
    n = len(Y)
    if n < 100:
        return np.nan, np.nan, np.nan, 0
    sf = s / s.mean()
    y = Y / sf[:, None]
    pois = sampling_variance(Y, sf, noise) if noise == "poisson" else _mn_var(Y, s, sf)
    mu = y.mean(axis=0)
    sst = ((y - mu) ** 2).sum(axis=0)
    codes, uniq = pd.factorize(lab)
    Z = np.zeros((n, len(uniq)))
    Z[np.arange(n), codes] = 1.0
    nk = Z.sum(axis=0)
    ssb = ((Z.T @ y) ** 2 / nk[:, None]).sum(axis=0) - n * mu ** 2
    df_res = n - len(uniq)
    if df_res < 20:
        return np.nan, np.nan, np.nan, 0
    t = sst / (n - 1) - pois
    r = (sst - ssb) / df_res - pois
    m = (t > 0) & np.isfinite(t) & np.isfinite(r)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan, int(m.sum())
    T = t[m].sum()
    return (float(1 - np.clip(r[m], 0, None).sum() / T), float(1 - r[m].sum() / T),
            float((r[m] < 0).mean()), int(m.sum()))


def _mn_var(Y, s, sf):
    """Multinomial sampling variance when the size factor comes from totals s over ALL genes
    (Y may hold a subset of genes)."""
    t = np.maximum(s, 2)[:, None]
    return (Y * (t - Y) / (t - 1) / (sf ** 2)[:, None]).mean(axis=0)


def ladder(xy, own, Y, seed, tgt, nb_labels=None, k_nb=20, n_groups=12, n_init=5, clip=True,
           noise="poisson", tot=None, n_perm=1):
    """Run S0, S1, S3 for the queried cells tgt.

    own        labels defining the S1 classes (each cell's own identity)
    nb_labels  labels entering neighbor compositions (default: own)
    tgt        boolean mask of queried cells; moves with the cell under the S3 permutation
    tot        per-cell totals for size factors (default: Y.sum(1))
    n_perm     S1 is averaged over this many within-label permutations. S1 is the only rung the own
               label enters, and a single permutation is unbiased but noisy (of the order of the own-label
               effect in a section of ~1,500 targets). The first permutation is the one used with n_perm=1,
               so n_perm=1 reproduces earlier results exactly. The neighborhood groups are shared by all
               permutations, so each extra permutation costs one R2 evaluation.
    Returns dict S0, S1, S3, identity, program, and negfrac_<rung> (share of genes clipped at each rung)."""
    own = np.asarray(own)
    nbl = own if nb_labels is None else np.asarray(nb_labels)
    tgt = np.asarray(tgt, bool)
    n = len(own)
    tot = (Y.sum(1) if tot is None else np.asarray(tot)).astype(np.float64)
    codes, uniq = pd.factorize(nbl)
    s1_codes = pd.factorize(pd.Series(own) + "|" + pd.Series(tgt.astype(str)))[0]
    nbr = cKDTree(xy).query(xy, k=k_nb + 1)[1][:, 1:]
    out, cent0, grp0, P0, M0 = {}, None, None, None, None
    for ri, rung in enumerate(["S0", "S1", "S3"]):
        rng = np.random.default_rng([seed, ri + 11])
        if rung == "S0":
            li = ei = np.arange(n)
        elif rung == "S1":
            li, ei = np.arange(n), permute_within(s1_codes, rng)
        else:
            li = ei = rng.permutation(n)
        c_l, tg_l = codes[li], tgt[li]
        P = np.where(tg_l)[0]
        if len(P) < 100:
            out[rung] = np.nan
            continue
        M = np.zeros((len(P), len(uniq)))
        nb = c_l[nbr[P]]
        for j in range(nb.shape[1]):
            np.add.at(M, (np.arange(len(P)), nb[:, j]), 1.0)
        M = np.sqrt(M / M.sum(1, keepdims=True))
        if rung == "S0":
            grp, cent0 = kmeans_best(M, n_groups, np.random.default_rng([seed, 999]), n_init=n_init)
            grp0, P0, M0 = grp, P, M
        elif P0 is not None and np.array_equal(P, P0) and np.array_equal(M, M0):
            grp = grp0
        elif cent0 is not None:
            grp, _ = kmeans(M, n_groups, None, init=cent0)
        else:
            grp, _ = kmeans_best(M, n_groups, np.random.default_rng([seed, 999]), n_init=n_init)
        j = 0 if clip else 1
        res = r2_micro(Y[ei[P]], tot[ei[P]], grp, noise=noise)
        v, nf = res[j], res[2]
        if rung == "S1" and n_perm > 1:
            vals, nfs = [v], [nf]
            for r in range(1, n_perm):
                e_r = permute_within(s1_codes, np.random.default_rng([seed, ri + 11, r]))
                rr = r2_micro(Y[e_r[P]], tot[e_r[P]], grp, noise=noise)
                vals.append(rr[j]); nfs.append(rr[2])
            v, nf = float(np.nanmean(vals)), float(np.nanmean(nfs))
        out[rung] = v
        out[f"negfrac_{rung}"] = nf       # share of genes whose residual variance was negative (clipped at 0)
    out["identity"] = out["S1"] - out["S3"]
    out["program"] = out["S0"] - out["S1"]
    return out
