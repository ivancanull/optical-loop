from argparse import Namespace
from pathlib import Path

import pytest
import optical_loop
import opticalloop
import opticalloop.config as config
from opticalloop.result import SimulationResult


def test_layer_command_uses_layer_simulator(monkeypatch, capsys) -> None:
    captured = {}

    class FakeLayerSimulator:
        def __init__(self, *, layer, architecture, backend, cache):
            captured["layer"] = layer
            captured["architecture"] = architecture
            captured["backend"] = backend
            captured["cache"] = cache

        def run(self):
            return SimulationResult(
                cycles=42,
                latency_s=1e-6,
                energy_j=2e-6,
                area_mm2=3e-3,
                tops_per_w=4.0,
                energy_breakdown={},
                mapping_text="for P in [0:1)\n  << Compute >>",
            )

    monkeypatch.setattr(optical_loop, "LayerSimulator", FakeLayerSimulator)
    args = Namespace(
        network="alexnet",
        workload=None,
        layer="alexnet/0",
        arch="proposed_mrr_optical_shift_add",
        var=[],
        system="fetch_all_lpddr4",
        tiles=1,
        pes=1,
        cols=100,
        rows=12,
        voltage_dac_resolution=1,
        scaling='"aggressive"',
        max_utilization=False,
        cache_results_dir=None,
        cache_save_name="deapcnns",
        cache_output_postfix="_1bit_input_osa",
        show_mapping=False,
    )

    optical_loop._run_layer(args)

    output = capsys.readouterr().out
    assert "OpticalLoop Timeloop-backed layer result" in output
    assert captured["layer"].layer_path == "alexnet/0"
    assert captured["architecture"].architecture_key == "T1, P1, C100, R12"
    assert captured["architecture"].macro == "proposed_mrr_optical_shift_add"


def test_layer_command_accepts_generic_architecture_variables(monkeypatch, capsys) -> None:
    captured = {}

    class FakeLayerSimulator:
        def __init__(self, *, layer, architecture, backend, cache):
            captured["layer"] = layer
            captured["architecture"] = architecture
            captured["cache"] = cache

        def run(self):
            return SimulationResult(
                cycles=7,
                latency_s=4e-9,
                energy_j=8e-9,
                energy_breakdown={},
                mapping_text="timeloop mapping text",
            )

    monkeypatch.setattr(optical_loop, "LayerSimulator", FakeLayerSimulator)
    args = Namespace(
        network=None,
        workload="deap_deepbench/bench0",
        layer=None,
        arch="deap_cnns",
        var=["N_COLUMNS=100", "N_ROWS=12", "N_Conv=1", "USE_FAST=true"],
        system="fetch_all_lpddr4",
        tiles=None,
        pes=None,
        cols=None,
        rows=None,
        voltage_dac_resolution=1,
        scaling='"aggressive"',
        max_utilization=False,
        cache_results_dir=None,
        cache_save_name="deapcnns",
        cache_output_postfix="_1bit_input_osa",
        show_mapping=True,
    )

    optical_loop._run_layer(args)

    output = capsys.readouterr().out
    assert "Mapping:" in output
    assert "timeloop mapping text" in output
    assert captured["layer"].network == "deap_deepbench"
    assert captured["architecture"].macro == "deap_cnns"
    assert captured["architecture"].to_timeloop_variables() == {
        "N_COLUMNS": 100,
        "N_ROWS": 12,
        "N_Conv": 1,
        "USE_FAST": True,
    }


def test_layer_command_rejects_malformed_variable() -> None:
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        optical_loop._parse_variable_assignments(["N_COLUMNS"])


def test_artifact_source_must_stay_inside_repo() -> None:
    with pytest.raises(SystemExit, match="must stay inside"):
        optical_loop._ensure_path_within_repo(
            Path("..") / "results",
            label="--artifact-source-results-dir",
        )


def test_public_api_is_timeloop_focused() -> None:
    assert hasattr(opticalloop, "TimeloopMacroConfig")
    assert hasattr(config, "TimeloopMacroConfig")
    assert not hasattr(opticalloop, "LinearLayerConfig")
    assert not hasattr(config, "LinearLayerConfig")
