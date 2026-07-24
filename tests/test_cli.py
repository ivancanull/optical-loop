import argparse
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
        arch="mrr_ws_osa",
        var=[],
        system="fetch_all_lpddr4",
        tiles=1,
        pes=1,
        cols=100,
        rows=12,
        front_mrr_slice_bits=1,
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
    assert captured["architecture"].macro == "mrr_ws_osa"


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


def test_public_api_is_timeloop_focused() -> None:
    assert hasattr(opticalloop, "TimeloopMacroConfig")
    assert hasattr(config, "TimeloopMacroConfig")
    assert not hasattr(opticalloop, "LinearLayerConfig")
    assert not hasattr(config, "LinearLayerConfig")

def test_public_cli_contains_only_current_workflows() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    optical_loop._add_layer_parser(subparsers)
    optical_loop._add_reproduce_parser(subparsers)
    optical_loop._add_multislice_parser(subparsers)
    optical_loop._add_accuracy_parser(subparsers)
    optical_loop._add_optimize_mapping_parser(subparsers)

    command_action = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(command_action.choices) == {
        "layer", "reproduce", "multislice", "accuracy", "optimize-mapping"
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["rosa"])



def test_reproduction_clis_accept_bounded_batches() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    optical_loop._add_reproduce_parser(subparsers)
    optical_loop._add_multislice_parser(subparsers)

    reproduce = parser.parse_args(["reproduce", "full", "--max-jobs", "256"])
    multislice = parser.parse_args(["multislice", "full", "--max-jobs", "128"])
    assert reproduce.max_jobs == 256
    assert multislice.max_jobs == 128

def test_incomplete_batch_stops_before_analysis(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        "{\"status\": \"incomplete\", \"successful_jobs\": 256, "
        "\"expected_jobs\": 42240, \"remaining_jobs\": 41984}"
    )

    assert optical_loop._report_incomplete_batch(run_dir)
    output = capsys.readouterr().out
    assert "256/42240 successful" in output
    assert "41984 pending" in output

def test_make_full_targets_distinguish_complete_and_bounded_runs() -> None:
    text = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    full = text.split("full: image", 1)[1].split("full-batch: image", 1)[0]
    full_batch = text.split("full-batch: image", 1)[1].split("multislice-smoke: image", 1)[0]
    multislice = text.split("multislice-full: image", 1)[1].split("multislice-full-batch: image", 1)[0]
    multislice_batch = text.split("multislice-full-batch: image", 1)[1]
    assert "--max-jobs" not in full
    assert "--max-jobs" in full_batch
    assert "--max-jobs" not in multislice
    assert "--max-jobs" in multislice_batch


def test_actions_run_one_layer_instead_of_research_smoke() -> None:
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/reproduction-smoke.yml"
    ).read_text()

    assert "reproduce smoke" not in workflow
    assert "multislice smoke" not in workflow
    assert "reproduce doctor" in workflow
    assert "multislice doctor" in workflow
    assert workflow.count("optical_loop.py layer") == 1
    assert "branches: [main]" in workflow
    assert "cancel-in-progress: true" in workflow
    assert (
        "optical_loop.py layer --arch mrr_ws_osa --workload alexnet/0 "
        "--tiles 1 --pes 16 --cols 8 --rows 8 "
        "--front-mrr-slice-bits 1 --show-mapping"
    ) in workflow
