"""Reading the input: counts, labels, coordinates and sections from an AnnData object or an .h5ad file.

For files, only the selected rows are read (h5py, row-selective for CSR and dense, column-streamed for
CSC), so datasets with millions of cells can be analyzed on a laptop. Counts are held as CSR int32.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

DEFAULT_EXCLUDE = r"(?i)^(blank|negcontrol|negcontrolprobe|negcontrolcodeword|notarget|unassigned|control|antisense)"


@dataclass
class Data:
    counts: sp.csr_matrix          # cells x genes, int32
    genes: np.ndarray
    label: np.ndarray              # published labels (str)
    section: np.ndarray            # analysis unit (section or crop)
    parent: np.ndarray             # physical section (for leave-one-section-out centroids and CV groups)
    condition: np.ndarray
    xy: np.ndarray
    truth: np.ndarray | None = None    # known true labels (semi-synthetic validation only), else None


def read_rows_h5(path, key, rows, col_mask=None, chunk=200_000):
    """Rows (sorted, unique) of an h5ad matrix ('X', 'raw/X' or 'layers/<name>') as CSR int32."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f[key]
        out = []
        if isinstance(g, h5py.Dataset):                                   # dense
            for i in range(0, len(rows), chunk):
                r = rows[i:i + chunk]
                lo, hi = int(r[0]), int(r[-1]) + 1
                blk = g[lo:hi][r - lo]
                blk = blk[:, col_mask] if col_mask is not None else blk
                out.append(sp.csr_matrix(np.rint(blk).astype(np.int32)))
            return sp.vstack(out).tocsr()
        enc = g.attrs.get("encoding-type", "csr_matrix")
        enc = enc.decode() if isinstance(enc, bytes) else enc
        shape = tuple(g.attrs["shape"])
        if enc == "csc_matrix":
            pos = np.full(shape[0], -1, np.int64)
            pos[rows] = np.arange(len(rows))
            cols = np.where(col_mask)[0] if col_mask is not None else np.arange(shape[1])
            newcol = np.full(shape[1], -1, np.int64)
            newcol[cols] = np.arange(len(cols))
            indptr = g["indptr"][:].astype(np.int64)
            rr, cc, dd = [], [], []
            j, budget = 0, 40_000_000
            while j < shape[1]:
                j2 = j + 1
                while j2 < shape[1] and indptr[j2 + 1] - indptr[j] <= budget:
                    j2 += 1
                lo, hi = indptr[j], indptr[j2]
                ind, dat = g["indices"][lo:hi], g["data"][lo:hi]
                colid = np.repeat(np.arange(j, j2), np.diff(indptr[j:j2 + 1]))
                keep = (pos[ind] >= 0) & (newcol[colid] >= 0)
                rr.append(pos[ind[keep]]); cc.append(newcol[colid[keep]])
                dd.append(np.rint(dat[keep]).astype(np.int32))
                j = j2
            return sp.csr_matrix((np.concatenate(dd), (np.concatenate(rr), np.concatenate(cc))),
                                 shape=(len(rows), len(cols)))
        if enc != "csr_matrix":
            raise ValueError(f"{key}: {enc} storage not supported")
        indptr = g["indptr"]
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]
            lo, hi = int(r[0]), int(r[-1]) + 1
            ip = indptr[lo:hi + 1].astype(np.int64)
            M = sp.csr_matrix((np.rint(g["data"][ip[0]:ip[-1]]).astype(np.int32), g["indices"][ip[0]:ip[-1]],
                               ip - ip[0]), shape=(hi - lo, shape[1]))[r - lo]
            out.append(M[:, col_mask] if col_mask is not None else M)
        return sp.vstack(out).tocsr()


def crop_rows(section, xy, rows, crop_cells, crops_per_section, seed):
    """Contiguous discs of crop_cells cells around well-separated centers (farthest-point sampling)."""
    rng = np.random.default_rng(seed)
    out_rows, out_id = [], []
    sec, P_all = section[rows], xy[rows]
    for s in np.unique(sec):                      # sorted, so that crops are those of the paper's runs
        ii = np.where(sec == s)[0]
        if len(ii) <= crop_cells:
            out_rows.append(rows[ii]); out_id += [str(s)] * len(ii)
            continue
        P = P_all[ii]
        tree = cKDTree(P)
        centres = [int(rng.integers(len(ii)))]
        d = np.linalg.norm(P - P[centres[0]], axis=1)
        for _ in range(crops_per_section - 1):
            centres.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(P - P[centres[-1]], axis=1))
        for k, c in enumerate(centres):
            nb = tree.query(P[c], k=crop_cells)[1]
            out_rows.append(rows[ii[np.sort(nb)]]); out_id += [f"{s}_c{k}"] * len(nb)
    return np.concatenate(out_rows), np.array(out_id)


def _coords(obs, obsm, spatial):
    if isinstance(spatial, (tuple, list)):
        return np.column_stack([obs[spatial[0]].astype(float), obs[spatial[1]].astype(float)])
    return np.asarray(obsm[spatial])[:, :2].astype(float)


def read_labels_file(path, label_key, obs_index):
    """Labels from an external table (csv/tsv, optionally gzipped, or parquet) whose first column is the cell ID
    (the AnnData obs index), aligned to obs_index. Lets alternative label sets be analyzed without writing into
    the data file."""
    if str(path).endswith(".parquet"):
        L = pd.read_parquet(path)
    else:
        sep = "\t" if re.search(r"\.tsv(\.gz)?$", str(path)) else ","
        L = pd.read_csv(path, sep=sep, index_col=0, dtype=str, keep_default_na=False)
    L.index = L.index.astype(str)
    if label_key not in L.columns:
        raise KeyError(f"{label_key!r} is not a column of {path}; columns: {list(L.columns)}")
    idx = pd.Index(np.asarray(obs_index).astype(str))
    missing = ~idx.isin(L.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} cells of the data have no row in {path}")
    return L.loc[idx, label_key].astype(str).to_numpy()


def _labels_columns(path):
    """Column names of a labels table, read as read_labels_file reads it (csv/tsv, optionally gzipped, or parquet)."""
    if str(path).endswith(".parquet"):
        import pyarrow.parquet as pq
        return list(pq.read_schema(path).names)
    sep = "\t" if re.search(r"\.tsv(\.gz)?$", str(path)) else ","
    return list(pd.read_csv(path, sep=sep, nrows=0, index_col=0).columns)


def load(adata, label_key, section_key, spatial=("x", "y"), counts_layer="X", condition_key=None,
         sections=None, crop_cells=0, crops_per_section=2, exclude_genes=DEFAULT_EXCLUDE, seed=0, labels_file=None,
         truth_key=None):
    """Return a Data object. `adata` is an AnnData object or a path to an .h5ad file.
    counts_layer: 'X', 'raw' (raw.X) or a layer name; must hold raw integer counts.
    spatial: (x_column, y_column) in obs, or an obsm key such as 'spatial'."""
    path = adata if isinstance(adata, str) else None
    if path:
        import anndata as ad
        A = ad.read_h5ad(path, backed="r")
    else:
        A = adata
    var = A.raw.var if (counts_layer == "raw" and A.raw is not None) else A.var
    genes = np.asarray(var.index.astype(str))
    keep_g = np.ones(len(genes), bool)
    if "feature_types" in var:
        keep_g &= var.feature_types.astype(str).str.contains("Gene Expression").to_numpy()
    if exclude_genes:
        keep_g &= ~pd.Series(genes).str.match(exclude_genes).to_numpy()
    obs = A.obs
    xy = _coords(obs, A.obsm, spatial)
    sec_all = obs[section_key].astype(str).to_numpy()
    rows = np.arange(len(obs)) if sections is None else np.where(np.isin(sec_all, list(sections)))[0]
    if crop_cells:
        rows, unit = crop_rows(sec_all, xy, rows, crop_cells, crops_per_section, seed)
    else:
        unit = sec_all[rows]
    urows, inv = np.unique(rows, return_inverse=True)
    if path:
        key = {"X": "X", "raw": "raw/X"}.get(counts_layer, f"layers/{counts_layer}")
        C = read_rows_h5(path, key, urows, col_mask=None if keep_g.all() else keep_g)[inv]
        A.file.close()
    else:
        M = A.raw.X if counts_layer == "raw" else (A.X if counts_layer == "X" else A.layers[counts_layer])
        M = M[urows][:, keep_g] if not keep_g.all() else M[urows]
        C = sp.csr_matrix(np.rint(M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.int32))[inv]
    C.data = C.data.astype(np.int32)
    cond = obs[condition_key].astype(str).to_numpy()[rows] if condition_key else np.array(["all"] * len(rows))
    lab_all = read_labels_file(labels_file, label_key, obs.index) if labels_file else obs[label_key].astype(str).to_numpy()
    truth = None
    if truth_key:                                  # from the label table if it has the column, else from obs
        if labels_file and truth_key in _labels_columns(labels_file):
            truth = read_labels_file(labels_file, truth_key, obs.index)[rows]
        else:
            truth = obs[truth_key].astype(str).to_numpy()[rows]
    return Data(counts=C.tocsr(), genes=genes[keep_g], label=lab_all[rows],
                section=np.asarray(unit).astype(str), parent=sec_all[rows], condition=cond, xy=xy[rows], truth=truth)


class TargetMask:
    """labels -> boolean mask of target cells; a regex (searched anywhere in the label), a list of labels,
    or a callable. Restricted to labels known to the analysis. Picklable (for parallel runs)."""

    def __init__(self, target, labels):
        self.known = set(map(str, labels))
        self.target = target
        if isinstance(target, str):
            rx = re.compile(target)
            self._set = {l for l in self.known if rx.search(l)}
        elif callable(target):
            self._set = None
        else:
            self._set = set(map(str, target)) & self.known

    def __call__(self, l):
        l = np.asarray(l).astype(str)
        if self._set is None:
            return np.asarray(self.target(l), bool) & np.isin(l, list(self.known))
        return np.isin(l, list(self._set))

    @property
    def target_labels(self):
        return sorted(self._set) if self._set is not None else None
