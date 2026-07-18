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
        bits = architecture.front_mrr_slice_bits
        cycles = 800 // bits
        dac = 0.01 * bits
        laser = 0.02
        energy = 0.1 + dac + laser
        front_dac = "input_dac" if "ws" in architecture.macro else "weight_dac"
        breakdown = {front_dac: dac, "laser": laser, "other": 0.1}
        if bits < 8 and "no_osa" not in architecture.macro:
            breakdown["delay_line"] = 0.0
        elif bits < 8:
            breakdown["digital_shift_add"] = 1e-6
        return SimulationResult(
            cycles=cycles,
            latency_s=cycles * 2e-10,
            energy_j=energy,
            energy_breakdown=breakdown,
            area_breakdown={"input_mrr": 1e-5, "weight_mrr": 1e-5},
            area_mm2=1e-3,
            compute=1000.0,
            cycle_seconds=2e-10,
            tops=1.0,
        )


class InterruptingBackend:
    def run_layer(self, layer, architecture) -> SimulationResult:
        raise KeyboardInterrupt


def test_front_mrr_slice_width_interface() -> None:
    config = MRRMacroConfig(n_tiles=1, n_pes=16, n_cols=8, n_rows=8, front_mrr_slice_bits=4)
    variables = config.to_timeloop_variables()
    assert variables["FRONT_MRR_SLICE_BITS"] == 4
    assert variables["FRONT_MRR_RADIX"] == 16
    assert variables["N_TEMPORAL_SLICES"] == 2
    assert variables["N_TEMPORAL_ACCUMULATIONS"] == 1
    with pytest.raises(ValueError, match="one of"):
        MRRMacroConfig(n_tiles=1, n_pes=16, n_cols=8, n_rows=8, front_mrr_slice_bits=3)
    timed = MRRMacroConfig(
        n_tiles=1, n_pes=16, n_cols=8, n_rows=8, frequency_hz=5e9
    )
    assert timed.to_timeloop_variables()["GLOBAL_CYCLE_SECONDS"] == pytest.approx(2e-10)


def test_multislice_manifest_expands_expected_jobs(manifest) -> None:
    smoke = manifest.jobs("smoke")
    full = manifest.jobs("full")
    assert len(smoke) == 72
    assert len(full) == 42240
    assert {job.front_mrr_slice_bits for job in full} == {1, 2, 4, 8}
    assert {(job.front_mrr_slice_bits, job.temporal_slices, job.radix) for job in smoke} == {
        (1, 8, 2), (2, 4, 4), (4, 2, 16), (8, 1, 256)
    }
    assert len({job.job_id for job in full}) == len(full)
    assert {(job.stationarity, job.sliced_operand) for job in full} == {
        ("WS", "input"), ("IS", "weight")
    }
    assert {job.macro for job in full} == {
        "mrr_ws_no_osa", "mrr_ws_osa", "mrr_is_no_osa", "mrr_is_osa"
    }


def test_four_canonical_architectures_encode_orthogonal_choices() -> None:
    root = Path(__file__).resolve().parents[1] / "workspace/models/arch/1_macro"
    macros = {
        name: (root / name / "arch.yaml").read_text()
        for name in ("mrr_ws_no_osa", "mrr_ws_osa", "mrr_is_no_osa", "mrr_is_osa")
    }
    assert "delay_line" in macros["mrr_ws_osa"]
    assert "delay_line" in macros["mrr_is_osa"]
    assert "digital_shift_add" not in macros["mrr_ws_osa"] + macros["mrr_is_osa"]
    assert "digital_shift_add" in macros["mrr_ws_no_osa"]
    assert "digital_shift_add" in macros["mrr_is_no_osa"]
    assert "name: delay_line" not in macros["mrr_ws_no_osa"] + macros["mrr_is_no_osa"]
    assert "*spatial_must_reuse_inputs" in macros["mrr_ws_osa"]
    assert "*spatial_must_reuse_weights" in macros["mrr_is_osa"]
    assert "resolution: FRONT_MRR_SLICE_BITS" in macros["mrr_is_osa"]
    assert "resolution: VOLTAGE_DAC_RESOLUTION" in macros["mrr_ws_osa"]
    for obsolete in (
        "proposed_mrr", "proposed_mrr_optical_shift_add",
        "proposed_mrr_wi_optical_shift_add",
        "proposed_mrr_1bit_input_delay_line",
        "proposed_mrr_1bit_input_delay_line_wi",
    ):
        assert not (root / obsolete).exists()


def test_is_mapper_has_explicit_output_channel_spatial_mapping() -> None:
    root = Path(__file__).resolve().parents[1] / "workspace/models"
    assert "timeout: MAPPER_TIMEOUT" in (root / "include/mapper.yaml").read_text()
    for macro in ("mrr_ws_no_osa", "mrr_ws_osa", "mrr_is_no_osa", "mrr_is_osa"):
        variables = (root / "arch/1_macro" / macro / "variables_free.yaml").read_text()
        assert "MAPPER_TIMEOUT: 100000" in variables
    for macro in ("mrr_is_no_osa", "mrr_is_osa"):
        architecture = (root / "arch/1_macro" / macro / "arch.yaml").read_text()
        assert "maximize_dims: [[M]]" in architecture
        photonic_pe = architecture.split("name: photonic_pe", 1)[1].split("- !Component", 1)[0]
        assert "*spatial_must_reuse_inputs" in photonic_pe


def test_dac_ert_contains_calibrated_slice_resolutions() -> None:
    path = Path(__file__).resolve().parents[1] / (
        "workspace/models/arch/1_macro/mrr_ws_osa/components/7nm_components.csv"
    )
    text = path.read_text()
    for bits, energy, area in ((1, "5.2", "3750"), (2, "10.4", "7500"),
                               (4, "20.8", "15000"), (8, "41.6", "30000")):
        assert f"7nm,aggressive,{bits},write|update,{energy},{area}" in text


def test_energy_models_and_optical_loss_scaling() -> None:
    model = SliceEnergyModel()
    assert [model.dac_factor(4, name) for name in (
        "optimistic_constant_symbol", "linear_bit", "conservative_walden"
    )] == [1.0, 4.0, 15.0]
    row = {
        "front_mrr_slice_bits": 4, "sliced_operand": "input", "temporal_slices": 2, "energy_j": 10.0,
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
    report = outputs["REPORT.md"].read_text()
    assert "Accuracy status: **NOT_MODELED**" in report
    assert "same-core best fixed" in report
    assert "Run tier: **SMOKE**" in report
    assert "not whole-network EDP results" in report
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["successful_jobs"] == 72


def test_interrupted_run_records_progress_for_resume(manifest, tmp_path: Path) -> None:
    runner = ReproductionRunner(manifest, tmp_path, backend=InterruptingBackend())
    with pytest.raises(KeyboardInterrupt):
        runner.run("smoke", workers=1)

    metadata_path = next(tmp_path.glob("*/run.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["successful_jobs"] == 0
    assert metadata["failed_jobs"] == 0
    assert metadata["remaining_jobs"] == 72


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
