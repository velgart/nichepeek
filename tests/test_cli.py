from nichepeek.cli import main


def test_simulate_run_report(tmp_path):
    f, out = str(tmp_path / "sim.h5ad"), str(tmp_path / "out")
    assert main(["simulate", f, "--n-sections", "2", "--n-cells", "2000", "--label-noise", "0.1"]) == 0
    assert main(["run", f, "--label-key", "label", "--target", "^F", "--section-key", "section",
                 "--counts-layer", "counts", "--p", "0.5", "--out", out]) == 0
    for name in ["sections.csv", "summary.csv", "result.json", "report.pdf"]:
        assert (tmp_path / "out" / name).exists(), name
    assert main(["report", out]) == 0
