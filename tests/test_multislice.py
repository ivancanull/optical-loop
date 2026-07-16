import json
from pathlib import Path

import pandas as pd
import pytest

from opticalloop.applications.rosa.multislice import (
    ACCURACY_STATUS,
    ASWMOptimizer,
    MultiSliceAnalyzer,
    ParetoState,
    SliceChoice,
    SliceEnergyModel,
    MultiSliceValidator,
)
from opticalloop.applications.rosa.reproduction import ExperimentManifest, ReproductionRunner
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.result import SimulationResult


@pytest.fixture
def manifest() -> ExperimentManifest:
    root = Path(__file__).resolve().parents[1]
    return ExperimentManifest(root / "examples/rosa/mb_osa_manifest.yaml", repo_root=root)


class MultiSliceFakeBackend:
    def run_layer(self, layer, architecture) -> SimulationResult:
        bits = architecture.input_slice_bits
        cycles = 800 // bits
        dac = 0.01 * bits
        laser = 0.02
        energy = 0.1 + dac + laser
        return SimulationResult(
            cycles=cycles,
            latency_s=cycles * 1e-9,
            energy_j=energy,
            energy_breakdown={"input_dac": dac, "laser": laser, "other": 0.1},
            area_mm2=1e-3,
            compute=1000.0,
            cycle_seconds=1e-9,
            tops=1.0,
        )


def test_architecture_slice_width_interface_and_legacy_alias() -> None:
    config = MRRMacroConfig(n_tiles=1, n_pes=16, n_cols=8, n_rows=8, input_slice_bits=4)
    variables = config.to_timeloop_variables()
    assert variables["INPUT_SLICE_BITS"] == 4
    assert variables["SLICE_RADIX"] == 16
    assert variables["N_TEMPORAL_SLICES"] == 2
    assert variables["N_DELAY_STAGES"] == 1
    with pytest.warns(DeprecationWarning):
        legacy = MRRMacroConfig(
            n_tiles=1, n_pes=16, n_cols=8, n_rows=8, voltage_dac_resolution=2
        )
    assert legacy.input_slice_bits == 2
    with pytest.raises(ValueError, match="one of"):
        MRRMacroConfig(n_tiles=1, n_pes=16, n_cols=8, n_rows=8, input_slice_bits=3)


def test_multislice_manifest_expands_expected_jobs(manifest) -> None:
    smoke = manifest.jobs("smoke")
    full = manifest.jobs("full")
    assert len(smoke) == 24
    assert len(full) == 14080
    assert {job.slice_bits for job in full} == {1, 2, 4, 8}
    assert {(job.slice_bits, job.temporal_slices, job.radix) for job in smoke} == {
        (1, 8, 2), (2, 4, 4), (4, 2, 16), (8, 1, 256)
    }
    assert len({job.job_id for job in full}) == len(full)


def test_energy_models_and_optical_loss_scaling() -> None:
    model = SliceEnergyModel()
    assert [model.dac_factor(4, name) for name in (
        "optimistic_constant_symbol", "linear_bit", "conservative_walden"
    )] == [1.0, 4.0, 15.0]
    row = {
        "slice_bits": 4, "temporal_slices": 2, "energy_j": 10.0,
        "energy_breakdown": {"input_dac": 4.0, "laser": 2.0, "other": 4.0},
    }
    assert model.adjust(row, "linear_bit", 0.0) == pytest.approx(10.0)
    assert model.adjust(row, "optimistic_constant_symbol", 0.0) == pytest.approx(7.0)
    assert model.adjust(row, "conservative_walden", 0.0) == pytest.approx(21.0)
    assert model.adjust(row, "linear_bit", 1.0) > 10.0


def test_exact_pareto_optimizer_finds_mixed_slice_optimum() -> None:
    choices = {
        "0": (
            SliceChoice("0", 1, 1.0, 8.0),
            SliceChoice("0", 2, 2.0, 4.0),
            SliceChoice("0", 4, 6.0, 2.0),
        ),
        "1": (
            SliceChoice("1", 1, 1.0, 8.0),
            SliceChoice("1", 2, 4.0, 4.0),
            SliceChoice("1", 4, 5.0, 2.0),
        ),
    }
    best, frontier = ASWMOptimizer().optimize(choices)
    brute_force = [
        ParetoState(a.energy_j + b.energy_j, a.latency_s + b.latency_s, (a, b))
        for a in choices["0"] for b in choices["1"]
    ]
    assert best.edp_j_s == min(state.edp_j_s for state in brute_force)
    assert all(
        not (other.energy_j <= state.energy_j and other.latency_s <= state.latency_s
             and (other.energy_j < state.energy_j or other.latency_s < state.latency_s))
        for state in frontier for other in frontier
    )


def test_exact_optimizer_uses_native_cycles_and_enforces_safety_cap() -> None:
    choices = {
        "0": tuple(
            SliceChoice("0", bits, float(bits), latency, cycles)
            for bits, latency, cycles in (
                (1, 0.1 + 0.2, 3),
                (2, 0.3, 3),
                (4, 0.1, 1),
            )
        ),
        "1": tuple(
            SliceChoice("1", bits, float(bits), 0.1 * cycles, cycles)
            for bits, cycles in ((1, 4), (2, 2), (4, 1))
        ),
    }
    _, frontier = ASWMOptimizer(frontier_limit=10).optimize(choices)
    assert len({state.cycles for state in frontier}) == len(frontier)
    with pytest.raises(RuntimeError, match="frontier exceeded"):
        ASWMOptimizer(frontier_limit=1).optimize(choices)


def test_multislice_smoke_analysis_and_accuracy_boundary(manifest, tmp_path: Path) -> None:
    run_dir = ReproductionRunner(
        manifest, tmp_path, backend=MultiSliceFakeBackend()
    ).run("smoke", workers=2)
    outputs = MultiSliceAnalyzer(manifest, run_dir).analyze()
    checks = pd.read_csv(outputs["validation.csv"])
    raw = pd.read_csv(outputs["layer_results.csv"])
    aswm = pd.read_csv(outputs["aswm_summary.csv"])
    assert checks.status.eq("PASS").all(), checks.to_string(index=False)
    assert raw.accuracy.isna().all()
    assert raw.accuracy_status.eq(ACCURACY_STATUS).all()
    assert aswm.accuracy.isna().all()
    assert "Accuracy status: **NOT_MODELED**" in outputs["REPORT.md"].read_text()
    assert "same-core best fixed" in outputs["REPORT.md"].read_text()
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["successful_jobs"] == 24


def test_validator_rejects_substituted_job_and_invalid_shape(manifest, tmp_path: Path) -> None:
    run_dir = ReproductionRunner(
        manifest, tmp_path, backend=MultiSliceFakeBackend()
    ).run("smoke", workers=1)
    analyzer = MultiSliceAnalyzer(manifest, run_dir)
    raw = analyzer.raw_dataframe()
    raw.loc[0, "job_id"] = "unexpected-job"
    raw.loc[1, "rows"] = int(raw.loc[1, "rows"]) + 1
    modeled = analyzer._modeled_rows(raw)
    fixed = analyzer._fixed_summary(modeled)
    selections, aswm, _ = analyzer._aswm_results(modeled)
    checks = MultiSliceValidator(manifest, run_dir).validate(
        raw, modeled, fixed, selections, aswm
    ).set_index("check")
    assert checks.loc["exact_expected_jobs", "status"] == "FAIL"
    assert checks.loc["architecture_constraints", "status"] == "FAIL"
