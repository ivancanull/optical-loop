import pandas as pd
import pytest

from opticalloop.result import SimulationResult
from opticalloop.workflow.results import (
    aggregate_metrics_csv,
    reconstruct_breakdown_csv,
    write_architecture_metrics_csv,
    write_results_csv,
)


def test_result_writers_and_aggregation_round_trip(tmp_path) -> None:
    results = [
        SimulationResult(
            cycles=10,
            cycle_seconds=1e-9,
            latency_s=10e-9,
            energy_j=2e-6,
            area_mm2=0.01,
            tops=1.0,
            tops_per_w=2.0,
            tops_per_mm2=3.0,
            compute=100.0,
            energy_breakdown={"adc": 1e-6, "laser": 1e-6},
            area_breakdown={"adc": 0.002, "laser": 0.003},
            metadata={
                "architecture": "T1, P1, C3, R4",
                "row_index": "macro_toy_workspace/models/workloads/toy/0.yaml_1tiles_1pes_3cols_4rows",
            },
        ),
        SimulationResult(
            cycles=20,
            cycle_seconds=1e-9,
            latency_s=20e-9,
            energy_j=3e-6,
            area_mm2=0.01,
            tops=1.5,
            tops_per_w=2.5,
            tops_per_mm2=3.5,
            compute=200.0,
            energy_breakdown={"adc": 3e-6},
            area_breakdown={"adc": 0.002},
            metadata={
                "architecture": "T1, P1, C3, R4",
                "row_index": "macro_toy_workspace/models/workloads/toy/1.yaml_1tiles_1pes_3cols_4rows",
            },
        ),
    ]

    breakdown = tmp_path / "breakdown.csv"
    reconstructed = tmp_path / "reconstructed.csv"
    aggregated = tmp_path / "aggregated.csv"
    metrics = tmp_path / "metrics.csv"

    write_results_csv(results, breakdown, group_components=False)
    reconstruct_breakdown_csv(breakdown, reconstructed)
    aggregate_metrics_csv(reconstructed, aggregated)
    write_architecture_metrics_csv(results, metrics)

    aggregated_df = pd.read_csv(aggregated)
    assert len(aggregated_df) == 1
    assert aggregated_df.loc[0, "Energy"] == 5e-6
    assert aggregated_df.loc[0, "Cycles"] == 30
    assert aggregated_df.loc[0, "Energy_per_MAC"] == pytest.approx(5e-6 / 300.0)

    metrics_df = pd.read_csv(metrics)
    assert metrics_df.loc[0, "Architecture"] == "T1, P1, C3, R4"
    assert metrics_df.loc[0, "EDP"] == pytest.approx(5e-6 * 30e-9)
