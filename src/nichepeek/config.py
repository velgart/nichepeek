"""Settings of a diagnostic run. Defaults implement the recommended protocol: count splitting, target set
fixed by the published labels, channel arms, and the membership channel."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    p: tuple = (0.5, 0.25, 0.12)          # thinning levels (fractions of the counts the ladder sees)
    split: bool = True                    # count splitting: labels from half A, ladder on half B
    fixed_targets: bool = True            # queried cells = published targets in every arm
    channels: bool = True                 # own-label (T) and neighbor (N) channel arms
    membership: bool = True               # arms with targets re-selected by the re-called labels
    ncem: bool = False                    # also score every arm with the NCEM-style niche regression
    ncem_genes: int = 200                 # NCEM on the N genes with the highest mean count in published targets
                                          # (same genes in every arm); 0 = all genes with biological variance
    k: int = 20                           # neighbors
    n_groups: int = 12                    # neighborhood groups (k-means)
    n_init: int = 5
    n_boot: int = 3                       # Poisson resamples for the native flip rate
    n_splits: int = 1                     # independent random replicates (split, thinning) averaged per section
    n_perm: int = 1                       # within-label permutations averaged at the S1 rung (1 = original ladder)
    noise: str = "multinomial"            # sampling variance of size-normalized counts: multinomial (correct when
                                          # size factors come from the same counts) | poisson (kept for comparison)
    clip: bool = True                     # clipped R2 (as in the paper) or unclipped
    min_label_cells: int = 50
    centroid_sample: int = 3000           # cells per (section, label) for the default labeler
    seed: int = 0
    context_tests: bool = True
    min_targets: int = 50                 # sections (or crops) with fewer published target cells are skipped with a
                                          # message; their ladder is unstable
    min_depth: float = 10.0               # summaries: thinning levels whose effective depth (counts per cell the
                                          # ladder sees) is below this are excluded from slopes and lower bounds
                                          # and flagged; channel signs are unreliable there (simulations)
    extra: dict = field(default_factory=dict)

    def key(self):
        """Hash of every setting that changes per-section results (for resuming runs)."""
        d = asdict(self)
        d.pop("context_tests")
        d.pop("min_depth")                # summaries only; does not change per-section results
        d.pop("min_targets", None)        # selects sections; does not change any section's results
        if d.get("n_perm", 1) == 1:        # n_perm = 1 leaves the cache key as without it
            d.pop("n_perm")
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]
