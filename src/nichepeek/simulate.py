"""Synthetic tissue with known cell types and a planted neighborhood program (as in the accompanying
paper's simulations). Used for tests, examples and checking the diagnostic against truth."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

NONTGT = ["Epi", "SMC", "Imm", "Mac", "Endo", "Neur"]
TGT = ["FA", "FA2", "FB", "FB2"]
TYPES = NONTGT + TGT


def _grf(xy, length, rng, n_modes=40):
    k = rng.normal(0, 1 / length, (n_modes, 2))
    ph = rng.uniform(0, 2 * np.pi, n_modes)
    f = np.cos(xy @ k.T + ph).sum(1)
    return (f - f.mean()) / f.std()


def profiles(G, d_identity, d_sub, rng):
    base = rng.normal(0.0, 0.6, G)
    prof = {}
    for j, t in enumerate(NONTGT):
        p = base.copy(); p[12 * j:12 * (j + 1)] += 2.5; prof[t] = p
    fib, fam, sub = np.arange(72, 84), np.arange(84, 104), np.arange(104, 112)
    for t in TGT:
        p = base.copy(); p[fib] += 2.0
        p[fam[:10]] += d_identity if t in ("FA", "FA2") else -d_identity
        p[fam[10:]] += -d_identity if t in ("FA", "FA2") else d_identity
        p[sub] += d_sub if t in ("FA", "FB") else -d_sub
        prof[t] = p
    return prof, np.arange(112, 132)


def section(rng, n_cells=6000, density=0.0025, G=150, d_identity=1.0, d_sub=0.6, alpha=0.5, depth=70,
            segregation=1.5, target_weight=0.12, sigma_cell=0.3):
    """One synthetic section: coordinates, true labels, counts (cells x G)."""
    L = np.sqrt(n_cells / density)
    xy = rng.uniform(0, L, (n_cells, 2))
    u = xy[:, 1] / L + 0.15 * _grf(xy, 0.35 * L, rng)
    v, w = _grf(xy, 0.15 * L, rng), _grf(xy, 0.12 * L, rng)
    s = segregation
    logit = {"Epi": 12 * (0.35 - u), "SMC": 12 * (u - 0.75), "Imm": 1.2 * v - 0.5, "Mac": 1.5 * v - 1.0,
             "Endo": np.zeros(n_cells) - 0.8, "Neur": np.zeros(n_cells) - 1.8,
             "FA": s * (0.5 - u) * 4 + 0.6 * s * w, "FA2": s * (0.5 - u) * 4 - 0.6 * s * w,
             "FB": s * (u - 0.5) * 4 + 0.6 * s * w, "FB2": s * (u - 0.5) * 4 - 0.6 * s * w}
    W = np.column_stack([np.exp(np.clip(logit[t], -20, 20)) for t in TYPES])
    W[:, len(NONTGT):] *= target_weight
    W /= W.sum(1, keepdims=True)
    lab = np.array(TYPES)[(rng.random(n_cells)[:, None] > W.cumsum(1)).sum(1)]
    prof, prog = profiles(G, d_identity, d_sub, rng)
    P = np.vstack([prof[t] for t in lab])
    tgt = np.isin(lab, TGT)
    nbr = cKDTree(xy).query(xy, k=21)[1][:, 1:]
    infl = np.isin(lab[nbr], ["Imm", "Mac"]).mean(1)
    h = (infl - infl[tgt].mean()) / (infl[tgt].std() + 1e-9)
    P[np.ix_(tgt, prog)] += alpha * h[tgt, None]
    P = P + rng.normal(0, sigma_cell, P.shape)
    rate = np.exp(P); rate /= rate.sum(1, keepdims=True)
    Y = rng.poisson(rate * (depth * rng.lognormal(0, 0.3, n_cells))[:, None])
    return xy, lab, Y


def make_anndata(n_sections=4, seed=0, label_noise=0.0, **kw):
    """AnnData with counts in layers['counts'] and X, obs: section, x, y, true_label and 'label'
    (true labels, optionally with a fraction label_noise swapped within target families to mimic
    published labels with errors)."""
    import anndata as ad
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    Xs, obs = [], []
    for i in range(n_sections):
        xy, lab, Y = section(rng, **kw)
        pub = lab.copy()
        if label_noise:
            swap = {"FA": "FA2", "FA2": "FA", "FB": "FB2", "FB2": "FB"}
            m = np.isin(lab, TGT) & (rng.random(len(lab)) < label_noise)
            pub[m] = [swap[x] for x in lab[m]]
        Xs.append(Y)
        obs.append(pd.DataFrame(dict(section=f"s{i}", x=xy[:, 0], y=xy[:, 1], true_label=lab, label=pub)))
    X = sp.csr_matrix(np.vstack(Xs).astype(np.float32))
    o = pd.concat(obs, ignore_index=True)
    o.index = o.index.astype(str)
    A = ad.AnnData(X=X, obs=o)
    A.var_names = [f"g{j}" for j in range(X.shape[1])]
    A.layers["counts"] = X.copy()
    return A
