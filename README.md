# nichepeek

**See what cell-type label error does to your spatial niche analysis.**

Niche analyses ask how much of a cell's expression depends on its neighborhood *beyond its cell type*.
They condition on cell-type labels, and those labels are imperfect. `nichepeek` measures how label error
acts on such estimates in your data, with a design that removes two artifacts of the usual design:

* **Circularity.** Labels are called from the same counts the niche model is fitted to. Count splitting
  (labels from one binomial half of every cell's counts, the model scored on the other half) removes it.
* **Target membership.** When the population under study is selected by error-prone labels, label
  changes also change *who* is analyzed. Fixing the target set by the published labels removes it.

Label error then acts through three channels, reported separately:

| channel | what changes | effect found in the accompanying paper |
|---|---|---|
| own label (T) | the label of each target cell (its identity) | inflates the neighborhood share; grows with identity |
| neighbor (N) | the labels of its neighbors (its niche) | deflates the neighborhood share, usually weakly |
| membership (M) | who counts as a target | either sign; an artifact of label-selected populations |

The neighbor channel moves the neighborhood share little, but it can matter for single niche calls: in the paper's
semi-synthetic test with Niche-DE, a labeler's systematic errors reversed niche claims mostly through the neighbors'
labels, whereas errors spread at random over subtypes reversed none. Thinning measures errors that behave like
counting noise, not a labeler's systematic errors; agreement with a second annotation that errs differently is
the check for those.

## Install

```bash
git clone https://github.com/velgart/nichepeek && cd nichepeek && pip install .      # PyPI release to follow
```

## Quick start

```python
import nichepeek as npk
res = npk.diagnose("my_data.h5ad",               # or an AnnData object
                  label_key="celltype",         # published labels
                  target="^Fib",                # regex, list of labels or callable
                  section_key="sample",
                  spatial="spatial",            # obsm key, or ("x", "y") obs columns
                  counts_layer="counts",        # raw integer counts: "X", "raw" or a layer
                  out_dir="nichepeek_out", n_jobs=4)
print(res)                                      # plain-language summary
res.summary                                     # one row per thinning level
res.sections                                    # every arm and quantity per section
```

or from the shell:

```bash
nichepeek simulate toy.h5ad --label-noise 0.15          # a dataset with known truth
nichepeek run toy.h5ad --label-key label --target '^F' --section-key section \
    --counts-layer counts --out toy_out --n-jobs 4
```

`out_dir` receives `sections.csv`, `summary.csv`, `result.json` and a one-page `report.pdf`.
Per-section results are cached, so an interrupted run resumes where it stopped.

## What the output means

* `L_T_program` (own-label channel) > 0: label error in the targets' own labels inflates the neighborhood share.
* `L_program` (net), `L_N_program`, `L_M_program`: the net label effect and the other channels.
* `circ_*_native`: the circularity a same-count pipeline carries at your depth.
* `first_level_T` (in `result.json`): the own-label response per added flip at the first thinning level,
  times the native flip rate. Its **sign** was right in every independent unit of semi-synthetic data
  built from real tissue; its **magnitude** is first-order: the native flip rate understates the error rate, and a large first
  step overstates the response in deep data. (`lower_bound_T`, slope x flip, is kept so that the result tables deposited with
  the paper match; it is not a bound.)
* `clip_frac_S0`, `clip_frac_S1`: share of genes whose residual variance was negative and set to zero.
* Context tests: whether the neighborhood predicts *which* cells change label (usually barely) and
  whether changes *land* on context-typical labels (usually yes).
* `toward_context_all`, `n_flips_all`: the destination test with neighborhood profiles of every
  label, built from all cells. It also covers label changes that leave the target set, and target sets of a
  single label, where the original test (profiles from the targets only) is empty.
* `seconds`, `peak_memory_gb`: run time and the peak memory of the largest process.

**Define the targets as two or more labels.** The own-label channel moves variance between a cell's identity
(its label among the targets) and the neighborhood. A single target label has no identity among the targets,
so the channel cannot measure a bias there; the report says so instead of giving an estimate. The same holds,
less sharply, when target identity is low (below about 0.02): the report gives the value but flags it as
uninformative. A direction ("increased") is reported only when at least three quarters of the sections agree
and there are at least four of them. Sections or crops with fewer than `min_targets` (default 50) target cells
are skipped and listed (`--min-targets 0` keeps them).

The diagnostic gives the **direction** of the bias and a first-order magnitude. It does not correct your
estimates; it tells you whether to trust them and how to design around the problem.

## A magnitude: small first step and calibration

For a magnitude, run with a small first thinning step and calibrate:

```bash
nichepeek run data.h5ad ... --p 0.8,0.65,0.5 --calibrate            # or afterwards:
nichepeek calibrate results/run                                     # uses the packaged calibration table
nichepeek calibrate results/run --table my_calibration.csv          # or your own
```

The calibration divides by the estimate/measured ratio measured on semi-synthetic data built from real
tissue at the same first step and the nearest depth, and reports the range over all matching
settings. Runs outside the calibrated depth range are flagged. Build a table from your own semi-synthetic runs
with `make_calibration_table.py` (in the analysis scripts deposited with the paper; any table with the documented
columns works, see `nichepeek.calibrate`).

The truth labels of the semi-synthetic data have errors of their own. These shrink the measured bias, so the
uncorrected calibration tends to understate the own-label bias. Where the table has a truth-corrected divisor
(from truth-side extrapolation; paper scripts), `calibrate` reports the corrected value beside the
uncorrected one, and the two bracket the bias. Where the correction was not reliable, the report says so.

## Settings worth knowing

* `noise="multinomial"` (default): the sampling variance removed from size-normalized counts. Because
  size factors are computed from the same counts, counts are multinomial given the cell total; the Poisson
  form (`noise="poisson"`) overstates the noise by 1 / (1 - share of the gene) and is kept for comparison.
* `ncem=True, ncem_genes=200`: also score a standard niche regression (own label + neighbor composition),
  on the 200 genes most expressed in the targets, identical across arms. Label error acts through the
  same channels there, but regression is more sensitive to errors in neighbor labels.
* Channel signs are not reliable below ~10 counts per cell (in the ladder half).

## Large datasets

Files are read row-selectively. For sections with more than ~150k cells (or panels of thousands of
genes) analyze spatial crops: `crop_cells=30000, crops_per_section=2`. Use `n_jobs` to run sections in
parallel (memory grows with the number of workers). On macOS and Windows, a script that calls `diagnose` with
`n_jobs > 1` needs the usual `if __name__ == "__main__":` guard, because worker processes re-import it.

## Your own labeler

By default labels are re-called by a nearest-centroid classifier (leave-one-section-out centroids).
To test the labeling you actually use, pass `labeler=f` with `f(Y, section) -> labels`, applied to
half-counts and thinned counts.

## Output columns and the paper's notation

In column and key names, `program` is the neighborhood share $V$ of the paper; the names are kept so that saved runs stay readable.

Per-section columns are named `<arm>_<quantity>` (e.g. `a_program`, `T0.25_identity`) and label effects
`L_<channel>_<quantity>_<p>`. The third rung of the ladder, cells relocated at random, is S2 in the paper and keeps
the column name `S3` here (e.g. `a_S3`), so that the result tables deposited with the paper match.

## Citation

Elgart V, Loscalzo J. Cell-type label errors inflate neighborhood effects and can reverse niche claims in spatial transcriptomics (preprint;
identifier to be added). Software: nichepeek 0.3.5, doi:10.5281/zenodo.23987423 (https://doi.org/10.5281/zenodo.23987423).
