"""The report's reading is fitted to the page (smaller font, then a longer page), never cut off."""
import json
import types

import numpy as np
import pandas as pd


def _result(n_lines):
    S = pd.DataFrame({"p": [0.8, 0.65, 0.5], "effective_depth": [117.0, 95.0, 73.0],
                      "L_program": [0.002, 0.006, 0.01], "L_T_program": [0.003, 0.006, 0.01]})
    o = dict(n_sections=12, median_counts=293, flip_native=0.1, acc_c1_vs_published=0.83, identity_published=0.17,
             program_published=0.065, seconds=60, peak_memory_gb=2.0, ladder_depth=147.0)
    line = "Own-label channel at 0.65: +0.0061 (12/12 sections > 0). " * 3
    return types.SimpleNamespace(summary=S, overall=o, interpretation=[line] * n_lines)


def _text_bottom(fig):
    r = fig.canvas.get_renderer()
    return min(t.get_window_extent(renderer=r).y0 for t in fig.texts) / fig.dpi


def test_reading_fits_page():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nichepeek.report import fit_text
    for n in (3, 15, 60):
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.12, 0.62, 0.34, 0.28])
        top_in = (1 - 0.62 - 0.28) * fig.get_figheight()
        res = _result(n)
        fit_text(fig, "head line", res.interpretation, top=0.24)
        assert _text_bottom(fig) >= 0.24                      # nothing below the bottom margin
        p = ax.get_position()                                 # the panel keeps its distance from the top
        assert abs((1 - p.y1) * fig.get_figheight() - top_in) < 1e-6
        if n == 3:
            assert abs(fig.get_figheight() - 11.69) < 1e-9    # short readings keep the A4 page
        plt.close(fig)


def test_make_report_writes(tmp_path):
    from nichepeek.report import make_report
    p = make_report(_result(40), str(tmp_path / "r.png"))
    assert (tmp_path / "r.png").exists() and p.endswith("r.png")
