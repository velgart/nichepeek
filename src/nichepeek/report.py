"""One-page report (PDF or PNG) of a diagnostic run. The reading below the panels is fitted to the page
(smaller font, then a longer page) instead of running off the bottom."""
from __future__ import annotations

import textwrap

import numpy as np


def _wrap(head, lines, fs, base_fs=8.0, base_width=110):
    width = int(base_width * base_fs / fs)              # monospace: characters per line scale with 1 / font size
    return textwrap.fill(head, width) + "\n\n" + "\n".join(textwrap.fill(l, width) for l in lines)


def fit_text(fig, head, lines, top=0.24, x=0.06, bottom_in=0.25, sizes=(8, 7.5, 7, 6.5, 6)):
    """Place the reading below the panels so that none of it is cut off. The font shrinks from 8 to 6 pt
    (lines re-wrapped to the same width); if the text still does not fit, the page is extended downwards and
    everything above keeps its size and position relative to the top."""
    r = fig.canvas.get_renderer()
    H = fig.get_figheight()
    for fs in sizes:
        t = fig.text(x, top, _wrap(head, lines, fs), va="top", fontsize=fs, family="monospace")
        bb = t.get_window_extent(renderer=r)
        if bb.y0 / fig.dpi >= bottom_in:
            return t
        t.remove()
    fs = sizes[-1]
    t = fig.text(x, top, _wrap(head, lines, fs), va="top", fontsize=fs, family="monospace")
    need = bottom_in - t.get_window_extent(renderer=r).y0 / fig.dpi   # inches missing at the bottom
    t.remove()
    H2 = H + need
    for ax in fig.axes:                                  # keep every axis in inches, measured from the top
        p = ax.get_position()
        ax.set_position([p.x0, (p.y0 * H + need) / H2, p.width, p.height * H / H2])
    for txt in fig.texts:
        x0, y0 = txt.get_position()
        txt.set_position((x0, (y0 * H + need) / H2))
    if fig._suptitle is not None:
        x0, y0 = fig._suptitle.get_position()
        fig._suptitle.set_position((x0, (y0 * H + need) / H2))
    fig.set_figheight(H2)
    return fig.text(x, (top * H + need) / H2, _wrap(head, lines, fs), va="top", fontsize=fs, family="monospace")


def make_report(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S, o = result.summary, result.overall
    col = lambda c: S[c].to_numpy(dtype=float)          # plain arrays: matplotlib must not receive pandas objects
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("nichepeek: label-error diagnostic", fontsize=13, y=0.985)
    ax1 = fig.add_axes([0.12, 0.62, 0.34, 0.28])
    for c, lab, st in [("L_program", "net", "k-o"), ("L_T_program", "own label (T)", "C0-s"),
                       ("L_N_program", "neighbor (N)", "C1-^"), ("L_M_program", "membership (M)", "C2-v")]:
        if c in S:
            ax1.plot(col("effective_depth"), col(c), st, label=lab, ms=5)
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_xscale("log"); ax1.invert_xaxis()
    ax1.set_xlabel("effective depth (counts per cell)"); ax1.set_ylabel("label effect on neighborhood share")
    ax1.set_title("a  channels", loc="left", fontsize=10); ax1.legend(fontsize=7, frameon=False)
    ax2 = fig.add_axes([0.58, 0.62, 0.36, 0.28])
    if "circ_identity" in S:
        d = np.r_[o["ladder_depth"], col("effective_depth")]
        ci = np.r_[o["circ_identity_native"], col("circ_identity")]
        cp = np.r_[o["circ_program_native"], col("circ_program")]
        ax2.plot(d, ci, "C3-o", ms=5, label="identity"); ax2.plot(d, cp, "C4-s", ms=5, label="neighborhood share")
        ax2.axhline(0, color="grey", lw=0.8); ax2.set_xscale("log"); ax2.invert_xaxis(); ax2.legend(fontsize=7, frameon=False)
    ax2.set_xlabel("counts per cell"); ax2.set_ylabel("own half - other half")
    ax2.set_title("b  circularity", loc="left", fontsize=10)
    ax3 = fig.add_axes([0.12, 0.30, 0.34, 0.24])
    suf = ""                                          # all-label profiles when the target-only test is empty
    if ("toward_context" not in S or not np.isfinite(col("toward_context")).any()) and "toward_context_all" in S:
        suf = "_all"
    if f"toward_context{suf}" in S and np.isfinite(col(f"toward_context{suf}")).any():
        xs = np.arange(len(S))
        nul, lo, hi = col(f"toward_null{suf}"), col(f"toward_null{suf}_lo"), col(f"toward_null{suf}_hi")
        ax3.errorbar(xs, nul, yerr=np.vstack([nul - lo, hi - nul]), fmt="_", color="grey", capsize=3, label="null")
        ax3.plot(xs, col(f"toward_context{suf}"), "ko", label="observed" + (" (all-label profiles)" if suf else ""))
        ax3.set_xticks(xs, [f"p={p:g}" for p in S.p]); ax3.legend(fontsize=7, frameon=False)
    ax3.set_ylabel("label changes landing on\ncontext-typical labels"); ax3.set_title("c  destination test", loc="left", fontsize=10)
    ax4 = fig.add_axes([0.58, 0.30, 0.36, 0.24])
    if "auc_base" in S:
        ax4.plot(col("effective_depth"), col("auc_base"), "o-", color="grey", label="own label + depth")
        ax4.plot(col("effective_depth"), col("auc_full"), "ko-", label="+ neighborhood")
        ax4.set_xscale("log"); ax4.invert_xaxis(); ax4.legend(fontsize=7, frameon=False)
    ax4.set_xlabel("effective depth"); ax4.set_ylabel("AUC (which cells change label)")
    ax4.set_title("d  occurrence test", loc="left", fontsize=10)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for ax in (ax1, ax2, ax4):
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ticks = sorted({round(float(v)) for line in ax.get_lines()
                            if line.get_transform() != ax.get_yaxis_transform() for v in line.get_xdata()})
            ax.set_xticks([t for t in ticks if t > 0])
    head = (f"{o['n_sections']} sections; median {o['median_counts']:.0f} counts per cell; native flip "
            f"{o['flip_native']:.3f}; re-called vs published labels {o['acc_c1_vs_published']:.2f}; "
            f"published identity {o['identity_published']:.3f}, neighborhood share {o['program_published']:.3f}"
            + (f"; run time {o['seconds'] / 60:.1f} min, peak memory {o['peak_memory_gb']:.1f} GB (largest process)"
               if o.get("seconds") is not None and o.get("peak_memory_gb") is not None else ""))
    fit_text(fig, head, list(result.interpretation), top=0.24)
    fig.savefig(path)
    plt.close(fig)
    return path
