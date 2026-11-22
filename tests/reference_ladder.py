"""The ladder exactly as used for the paper (sim_tissue.ladder / scramble_ladder3): S1 permutes the
(label, expression) pair jointly. Kept here to test that the package's refactored ladder reproduces it."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from nichepeek.ladder import kmeans, kmeans_best, permute_within, r2_micro


def ladder_ref(xy, labels, Y, seed, is_target, k_nb=20, n_groups=12, n_init=5):
    n = len(labels)
    tot = Y.sum(1).astype(np.float64)
    codes, uniq = pd.factorize(labels)
    tgt = is_target(labels)
    s1_codes = pd.factorize(pd.Series(labels) + "|" + pd.Series(tgt.astype(str)))[0]
    nbr = cKDTree(xy).query(xy, k=k_nb + 1)[1][:, 1:]
    out, cent0, grp0, P0, M0 = {}, None, None, None, None
    for ri, rung in enumerate(["S0", "S1", "S3"]):
        rng = np.random.default_rng([seed, ri + 11])
        li = np.arange(n) if rung == "S0" else permute_within(s1_codes, rng) if rung == "S1" else rng.permutation(n)
        c_l, tg_l = codes[li], tgt[li]
        P = np.where(tg_l)[0]
        M = np.zeros((len(P), len(uniq)))
        nb = c_l[nbr[P]]
        for j in range(nb.shape[1]):
            np.add.at(M, (np.arange(len(P)), nb[:, j]), 1.0)
        M = np.sqrt(M / M.sum(1, keepdims=True))
        if rung == "S0":
            grp, cent0 = kmeans_best(M, n_groups, np.random.default_rng([seed, 999]), n_init=n_init)
            grp0, P0, M0 = grp, P, M
        elif np.array_equal(P, P0) and np.array_equal(M, M0):
            grp = grp0
        else:
            grp, _ = kmeans(M, n_groups, None, init=cent0)
        out[rung] = r2_micro(Y[li[P]], tot[li[P]], grp)[0]
    out["identity"] = out["S1"] - out["S3"]
    out["program"] = out["S0"] - out["S1"]
    return out
