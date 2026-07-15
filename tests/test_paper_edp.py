from pathlib import Path

import pytest

from opticalloop.applications.rosa.paper_edp import PaperEDPConfig, PaperEDPReproduction


def _reproduction() -> PaperEDPReproduction:
    root = Path(__file__).resolve().parents[1]
    return PaperEDPReproduction(
        PaperEDPConfig(root / "examples" / "rosa" / "paper_edp_data")
    )


def test_committed_edp_data_is_formula_consistent() -> None:
    checks = _reproduction().formula_checks()
    assert checks["passed"].all(), checks.to_string(index=False)


def test_six_workload_sweep_reproduces_eight_by_eight_winner() -> None:
    summary = _reproduction().headline_summary()
    assert summary["best_no_osa"] == "C8xR8"
    assert summary["optimized_vs_compact_reduction"] == pytest.approx(0.2775819265)
    assert summary["optimized_vs_deap_reduction"] == pytest.approx(0.4447485902)
    assert summary["osa_reduction_at_optimized_shape"] == pytest.approx(0.3139063019)
