import csv
import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import optical_loop
import opticalloop.applications.rosa.online_mapping as online_mapping_module
from opticalloop.accuracy.result import AccuracyResult
from opticalloop.applications.rosa.online_mapping import (
    CacheProvenance,
    MappingOptimizer,
    OptimizationConfig,
    PolicyEvaluator,
)
from opticalloop.result import SimulationResult


class FakeTimeloopBackend:
    def __init__(self) -> None:
        self.calls = []

    def run_layer(self, layer, architecture):
        self.calls.append((layer.layer_path, architecture.macro))
        is_mapping = architecture.macro == "mrr_is_osa"
        return SimulationResult(
            cycles=4 if is_mapping else 2,
            latency_s=4.0 if is_mapping else 2.0,
            energy_j=3.0 if is_mapping else 1.0,
            energy_breakdown={},
            source="fake-timeloop",
        )

    def run_batch(self, runs, n_jobs=None):
        return [self.run_layer(run.layer, run.architecture) for run in runs]


class FakeAccuracyBackend:
    def __init__(self, accuracy_by_is_count=None) -> None:
        self.calls = []
        self.accuracy_by_is_count = accuracy_by_is_count or (lambda count: 85.0)

    def run(self, experiment, policy):
        self.calls.append((policy, experiment.seeds))
        is_count = sum(value == "IS" for value in policy.stationarity.values())
        accuracy = float(self.accuracy_by_is_count(is_count))
        return AccuracyResult(
            network=experiment.network,
            scenario="hybrid",
            accuracies=tuple(accuracy for _ in experiment.seeds),
            losses=tuple(0.5 for _ in experiment.seeds),
            seeds=experiment.seeds,
            baseline_accuracy=86.5,
            source="fake-accuracy",
        )


def _config(root: Path, *, trials: int = 6, minimum_accuracy: float = 79.0):
    return OptimizationConfig(
        layer_manifest=root / "config/accuracy/resnet18_cifar10_layers.yaml",
        initial_policy=root / "config/accuracy/resnet18_hybrid_1bit.yaml",
        model_config=root / "unused-model.yaml",
        checkpoint=root / "unused-checkpoint.pth",
        onnsim_root=root / "unused-onnsim",
        trials=trials,
        minimum_accuracy=minimum_accuracy,
    )


def test_score_normalization_and_hard_accuracy_floor() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root)

    score, feasible = config.score(
        edp_j_s=5.0,
        accuracy=82.0,
        reference_edp_j_s=10.0,
        baseline_accuracy=86.0,
    )

    assert feasible is True
    assert score == pytest.approx(0.25 + 0.5 * (4.0 / 7.0))
    assert config.score(
        edp_j_s=1.0,
        accuracy=78.9,
        reference_edp_j_s=10.0,
        baseline_accuracy=86.0,
    ) == (None, False)


def test_config_rejects_invalid_weights() -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="sum to 1"):
        OptimizationConfig(
            layer_manifest=root / "layers.yaml",
            initial_policy=root / "policy.yaml",
            model_config=root / "model.yaml",
            checkpoint=root / "model.pth",
            onnsim_root=root / "onnsim",
            edp_weight=0.8,
            accuracy_weight=0.8,
        )


def test_evaluator_uses_network_edp_and_reuses_policy_cache(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    timeloop = FakeTimeloopBackend()
    accuracy = FakeAccuracyBackend()
    evaluator = PolicyEvaluator(
        _config(root), tmp_path, timeloop_backend=timeloop, accuracy_backend=accuracy
    )
    all_ws = {layer.layer_id: "WS" for layer in evaluator.manifest.layers}

    first = evaluator.evaluate(all_ws, (0,))
    second = evaluator.evaluate(all_ws, (0,))

    assert first["energy_j"] == 21.0
    assert first["latency_s"] == 42.0
    assert first["edp_j_s"] == 21.0 * 42.0
    assert second == first
    assert len(timeloop.calls) == 42
    assert len(accuracy.calls) == 1
    assert len(json.loads((tmp_path / "edp_lookup.json").read_text())) == 42


def test_evaluator_rejects_cache_from_different_provenance(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root)
    PolicyEvaluator(
        config,
        tmp_path,
        timeloop_backend=FakeTimeloopBackend(),
        accuracy_backend=FakeAccuracyBackend(),
    )

    with pytest.raises(ValueError, match="cache provenance differs"):
        PolicyEvaluator(
            replace(config, rows=config.rows * 2),
            tmp_path,
            timeloop_backend=FakeTimeloopBackend(),
            accuracy_backend=FakeAccuracyBackend(),
        )


def test_evaluator_rejects_legacy_cache_without_provenance(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "edp_lookup.json").write_text("{}")

    with pytest.raises(ValueError, match="lack provenance"):
        PolicyEvaluator(
            _config(root),
            tmp_path,
            timeloop_backend=FakeTimeloopBackend(),
            accuracy_backend=FakeAccuracyBackend(),
        )


def test_provenance_covers_execution_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    provenance = CacheProvenance.from_config(_config(root))

    assert set(provenance.payload["source_sha256"]) == {
        "online_mapping",
        "accuracy_runtime",
        "timeloop_backend",
        "simulation_result",
        "repository_config",
        "performance_models",
        "performance_helpers",
    }
    assert {
        "timeloopfe",
        "accelergy",
        "numpy",
        "torch",
        "torchvision",
        "cuda_version",
        "native_timeloop",
    } <= set(provenance.payload["tools"])
    assert set(provenance.payload["tools"]["native_timeloop"]) == {
        "mapper_path",
        "mapper_sha256",
        "shared_libraries",
    }


def test_provenance_hashes_every_performance_model_file(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "repository"
    module_path = repo_root / "applications/rosa/online_mapping.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# test module\n")
    models = repo_root / "workspace/models"
    models.mkdir(parents=True)
    template = models / "top.yaml.jinja2"
    template.write_text("architecture: version-1\n")
    (models / "components.csv").write_text("name,energy\nmrr,1.0\n")
    monkeypatch.setattr(online_mapping_module, "__file__", str(module_path))

    first = CacheProvenance.from_config(_config(repo_root))
    recorded = first.payload["source_sha256"]["performance_models"]

    assert recorded == CacheProvenance._digest_path(models)
    assert recorded != CacheProvenance._digest_path(
        models, source_files_only=True
    )

    template.write_text("architecture: version-2\n")
    second = CacheProvenance.from_config(_config(repo_root))

    assert second.payload["source_sha256"]["performance_models"] != recorded


def test_evaluator_rejects_changed_dataset_content(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = tmp_path / "runtime"
    dataset = runtime / "data"
    dataset.mkdir(parents=True)
    sample = dataset / "test_batch.bin"
    sample.write_bytes(b"dataset-v1")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("data_dir: data\n")
    config = replace(
        _config(root), model_config=model_config, onnsim_root=runtime
    )
    output_dir = tmp_path / "output"
    PolicyEvaluator(
        config,
        output_dir,
        timeloop_backend=FakeTimeloopBackend(),
        accuracy_backend=FakeAccuracyBackend(),
    )

    sample.write_bytes(b"dataset-v2")

    with pytest.raises(ValueError, match="cache provenance differs"):
        PolicyEvaluator(
            config,
            output_dir,
            timeloop_backend=FakeTimeloopBackend(),
            accuracy_backend=FakeAccuracyBackend(),
        )


def test_optimizer_warm_starts_and_finds_synthetic_all_is_optimum(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]

    class AllISBestTimeloop(FakeTimeloopBackend):
        def run_layer(self, layer, architecture):
            self.calls.append((layer.layer_path, architecture.macro))
            is_mapping = architecture.macro == "mrr_is_osa"
            return SimulationResult(
                cycles=1,
                latency_s=1.0,
                energy_j=1.0 if is_mapping else 3.0,
                energy_breakdown={},
                source="fake-timeloop",
            )

    evaluator = PolicyEvaluator(
        _config(root, trials=6),
        tmp_path,
        timeloop_backend=AllISBestTimeloop(),
        accuracy_backend=FakeAccuracyBackend(),
    )

    result = MappingOptimizer(evaluator.config, evaluator).optimize()
    trials = list(csv.DictReader((tmp_path / "trials.csv").open()))
    policy = yaml.safe_load((tmp_path / "best_policy.yaml").read_text())

    assert result["status"] == "success"
    assert result["selected_policy_key"] == "1" * 21
    assert [row["kind"] for row in trials[:3]] == ["predefined", "all_ws", "all_is"]
    assert len(trials) == 6
    assert all(item["stationarity"] == "IS" for item in policy["layers"].values())
    assert {
        "provenance.json",
        "edp_lookup.json",
        "policy_cache.json",
        "trials.csv",
        "best_policy.yaml",
        "best_result.json",
    } <= {path.name for path in tmp_path.iterdir()}


def test_confirmation_failure_returns_no_feasible_policy(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]

    class ConfirmationFailsAccuracy(FakeAccuracyBackend):
        def run(self, experiment, policy):
            self.calls.append((policy, experiment.seeds))
            accuracy = 85.0 if len(experiment.seeds) == 1 else 70.0
            return AccuracyResult(
                network=experiment.network,
                scenario="hybrid",
                accuracies=tuple(accuracy for _ in experiment.seeds),
                losses=tuple(0.5 for _ in experiment.seeds),
                seeds=experiment.seeds,
                baseline_accuracy=86.5,
                source="fake-accuracy",
            )

    evaluator = PolicyEvaluator(
        _config(root, trials=3),
        tmp_path,
        timeloop_backend=FakeTimeloopBackend(),
        accuracy_backend=ConfirmationFailsAccuracy(),
    )

    result = MappingOptimizer(evaluator.config, evaluator).optimize()

    assert result["status"] == "no_feasible_policy"
    assert result["phase"] == "confirmation"
    assert not (tmp_path / "best_policy.yaml").exists()
    assert json.loads((tmp_path / "best_result.json").read_text()) == result


def test_generated_candidates_are_complete_and_deterministic(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root, trials=8)

    def run(output):
        evaluator = PolicyEvaluator(
            config,
            output,
            timeloop_backend=FakeTimeloopBackend(),
            accuracy_backend=FakeAccuracyBackend(),
        )
        MappingOptimizer(config, evaluator).optimize()
        return list(csv.DictReader((output / "trials.csv").open()))

    first = run(tmp_path / "first")
    second = run(tmp_path / "second")

    assert [row["policy_key"] for row in first] == [
        row["policy_key"] for row in second
    ]
    assert all(len(row["policy_key"]) == 21 for row in first)
    assert all(set(row["policy_key"]) <= {"0", "1"} for row in first)


def test_optimize_mapping_cli_prints_result(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_run(config, output_dir):
        captured["config"] = config
        captured["output_dir"] = output_dir
        return {
            "status": "success",
            "trials": 3,
            "selected_policy_key": "0" * 21,
            "selected_score": 0.75,
            "selected_result": {
                "edp_j_s": 1e-9,
                "accuracy": 80.0,
            },
        }

    monkeypatch.setattr(optical_loop, "run_online_mapping", fake_run)
    args = Namespace(config=Path("config.yaml"), output_dir=tmp_path)

    optical_loop._run_optimize_mapping(args)

    output = capsys.readouterr().out
    assert "online WS/IS" in output
    assert "Accuracy:   80.0000%" in output
    assert captured == {"config": Path("config.yaml"), "output_dir": tmp_path}
