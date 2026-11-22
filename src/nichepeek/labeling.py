"""Re-calling cell-type labels from counts.

The diagnostic needs a labeler it can apply to arbitrary counts (half A of a split, thinned counts).
Any object with a `predict(Y, section) -> labels` method, or any callable `f(Y, section) -> labels`,
can be used. Y is a dense (cells x genes) integer array in the gene order of the input.

The default, CentroidLabeler, assigns each cell to the nearest label centroid by correlation of
log-normalized profiles, with centroids built from all OTHER sections (leave-one-section-out), so no
section is labeled with its own profile.
"""
from __future__ import annotations

import numpy as np


def norm_log(C, med):
    """log1p of counts scaled to a common total `med` per cell."""
    C = np.asarray(C)
    tot = np.maximum(C.sum(1), 1)[:, None]
    return np.log1p(C / tot * med)


def classify(L, cent, labels):
    """Nearest centroid by correlation distance (rows centerd)."""
    A = L - L.mean(1, keepdims=True)
    B = cent - cent.mean(1, keepdims=True)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return np.asarray(labels)[np.argmax(A @ B.T, axis=1)]


class CentroidLabeler:
    """Nearest-centroid labeler with leave-one-section-out centroids.

    fit(profiles) takes {(section, label): (sum of norm_log profiles, n cells)}; predict(Y, section)
    labels the cells of `section` with centroids from every other section."""

    def __init__(self, labels):
        self.labels = np.asarray(labels)
        self._sums, self._cnts = {}, {}

    def add(self, section, label, Lsum, n):
        self._sums[(section, label)] = Lsum
        self._cnts[(section, label)] = n

    def centroids(self, exclude):
        secs = {s for s, _ in self._sums}
        rows = []
        for l in self.labels:
            keys = [(s, l) for s in secs if s != exclude and (s, l) in self._sums]
            if not keys:                                   # label only present in the held-out section
                keys = [(s, l) for s in secs if (s, l) in self._sums]
            tot = sum(self._sums[k] for k in keys)
            rows.append(tot / max(1, sum(self._cnts[k] for k in keys)))
        return np.vstack(rows)

    def predict(self, Y, section, med=None):
        med = float(np.median(Y.sum(1))) if med is None else med
        return classify(norm_log(Y, med), self.centroids(section), self.labels)


class CallableLabeler:
    """Wraps a user function f(Y, section) -> labels."""

    def __init__(self, fn):
        self.fn = fn

    def predict(self, Y, section, med=None):
        return np.asarray(self.fn(Y, section)).astype(str)


def as_labeler(obj):
    if obj is None or hasattr(obj, "predict"):
        return obj
    if callable(obj):
        return CallableLabeler(obj)
    raise TypeError("labeler must have a predict(Y, section) method or be callable f(Y, section)")


class LogRegLabeler:
    """Alternative labeler: multinomial logistic regression on log-normalized counts, trained leave-one-section-
    out on the published labels (like the centroid labeler, no cell is labeled by a model trained on its own
    physical section). It is applied to half-depth and thinned counts, so it is trained on binomially thinned
    training cells (fraction drawn uniformly from `frac_range` per cell) rather than on full-depth profiles only.

    Build with LogRegLabeler.from_data(data, labels, keep); models are fitted lazily, one per held-out section,
    inside whichever process asks for them. Only a stratified sample of training cells is stored (at most
    `max_per` cells per section and label), so the object stays small enough to send to worker processes."""

    def __init__(self, labels, X, y, parent, scale, seed=0, max_per_label=400, frac_range=(0.05, 0.6), C=1.0):
        self.labels = np.asarray(labels)
        self.X, self.y, self.parent = X, np.asarray(y), np.asarray(parent)
        self.scale, self.seed, self.max_per_label = float(scale), int(seed), int(max_per_label)
        self.frac_range, self.C = tuple(frac_range), float(C)
        self._models = {}

    @classmethod
    def from_data(cls, data, labels, keep, seed=0, max_per=300, **kw):
        import pandas as pd
        rng = np.random.default_rng([seed, 17])
        idx = []
        for s in pd.unique(data.parent[keep]):
            for l in labels:
                ii = np.where(keep & (data.parent == s) & (data.label == l))[0]
                if len(ii):
                    idx.append(rng.choice(ii, min(len(ii), max_per), replace=False))
        idx = np.sort(np.concatenate(idx))
        X = data.counts[idx].tocsr()
        scale = float(np.median(np.asarray(X.sum(1)).ravel()))
        return cls(labels, X, data.label[idx], data.parent[idx], scale, seed=seed, **kw)

    def _features(self, Y):
        Y = np.asarray(Y, dtype=np.float32)
        tot = np.maximum(Y.sum(1, keepdims=True), 1.0)
        return np.log1p(Y / tot * self.scale)

    def _fit(self, section):
        from sklearn.linear_model import LogisticRegression
        import zlib
        rng = np.random.default_rng([self.seed, 23, zlib.crc32(str(section).encode())])
        tr = np.where(self.parent != section)[0]
        if len(np.unique(self.y[tr])) < len(np.unique(self.y)):      # a label only in the held-out section
            tr = np.arange(len(self.y))
        pick = []
        for l in np.unique(self.y[tr]):
            ii = tr[self.y[tr] == l]
            pick.append(rng.choice(ii, min(len(ii), self.max_per_label), replace=False))
        pick = np.concatenate(pick)
        Yt = self.X[pick].toarray()
        frac = rng.uniform(*self.frac_range, size=len(pick))[:, None]
        Yt = rng.binomial(Yt.astype(np.int64), frac)                 # thinned training cells
        m = LogisticRegression(C=self.C, max_iter=500)
        m.fit(self._features(Yt), self.y[pick])
        return m

    def predict(self, Y, section, med=None):
        if section not in self._models:
            self._models[section] = self._fit(section)
        return np.asarray(self._models[section].predict(self._features(Y))).astype(str)
