import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from opticalloop.accuracy import (
    AccuracyExperimentConfig,
    AccuracyResult,
    LayerManifest,
    LayerPolicy,
    MRRVariationConfig,
    attach_accuracy,
)
from opticalloop.accuracy.adapters import ONNSimAccuracyBackend


def _resnet_manifest(root: Path) -> LayerManifest:
    return LayerManifest.load(root / "config/accuracy/resnet18_cifar10_layers.yaml")


def _resnet_policy(root: Path, manifest: LayerManifest) -> LayerPolicy:
    return LayerPolicy.load(root / "config/accuracy/resnet18_hybrid_1bit.yaml", manifest)


def test_resnet_accuracy_manifest_covers_timeloop_workloads() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = _resnet_manifest(root)
    policy = _resnet_policy(root, manifest)
    workload_ids = {
        path.stem for path in (root / "workspace/models/workloads/resnet18").glob("*.yaml")
    }

    assert {layer.layer_id for layer in manifest.layers} == workload_ids
    assert manifest.module_for("00") == "conv1"
    assert manifest.module_for("20") == "classifier.1"
    assert policy.onnsim_mapping()["conv1"] is True
    assert policy.onnsim_mapping()["layer1.0.conv2"] is False


def test_policy_rejects_incomplete_layer_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = _resnet_manifest(root)

    with pytest.raises(ValueError, match="every layer"):
        LayerPolicy(
            manifest=manifest,
            stationarity={"00": "IS"},
            slice_bits={"00": 1},
        )


def test_local_resnet_preserves_layer_names_and_legacy_scale_shapes() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from opticalloop.accuracy.local_resnet import LocalModelConfig, load_checkpoint

    root = Path(__file__).resolve().parents[1]
    manifest = _resnet_manifest(root)
    model = LocalModelConfig(dropout=0.1).build()
    quantized_names = {
        name for name, module in model.named_modules()
        if hasattr(module, "weight_scale")
    }
    assert quantized_names == {layer.torch_module for layer in manifest.layers}

    legacy_state = dict(model.state_dict())
    for key, value in legacy_state.items():
        if "weight_scale" in key and value.shape == torch.Size([1]):
            legacy_state[key] = value.squeeze()
    assert load_checkpoint(model, legacy_state) is model


def test_local_mrr_matches_pinned_onnsim_fixture() -> None:
    torch = pytest.importorskip("torch")

    from opticalloop.accuracy.local_mrr import VariationAwareMRR

    torch.manual_seed(4)
    device = VariationAwareMRR()
    device.define_thermal_diffusion_behavior(0.05)
    device.define_DAC_variation_behavior(0.02)
    device.set_weight_mapping_range(0, 3, 1537, -1, 1)

    actual = device.simulate_variation(torch.tensor([-0.5, 0.0, 0.5]))
    expected = torch.tensor([-0.53654933, 0.00192058, 0.46946180])
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)


def test_thermal_noise_scaling_is_anchored_and_monotonic() -> None:
    variation = MRRVariationConfig(
        thermal_std=0.05, thermal_reference_bits=8, thermal_scaling_exponent=0.5
    )

    values = [variation.effective_thermal_std(bits) for bits in (1, 2, 4, 8)]

    assert values == sorted(values)
    assert len(set(values)) == 4
    assert values[-1] == 0.05


def test_thermal_noise_scaling_validates_inputs_and_preserves_zero() -> None:
    zero = MRRVariationConfig(thermal_std=0.0)
    assert all(zero.effective_thermal_std(bits) == 0.0 for bits in (1, 2, 4, 8))

    with pytest.raises(ValueError, match="exponent"):
        MRRVariationConfig(thermal_scaling_exponent=-0.1)
    with pytest.raises(ValueError, match="reference bits"):
        MRRVariationConfig(thermal_reference_bits=3)
    with pytest.raises(ValueError, match="slice bits"):
        zero.effective_thermal_std(3)


def test_accuracy_result_attaches_whole_network_constraint() -> None:
    result = AccuracyResult(
        network="resnet18",
        scenario="hybrid",
        accuracies=(84.0, 86.0),
        losses=(0.5, 0.4),
        seeds=(0, 1),
        baseline_accuracy=86.0,
        source="test",
    )

    joined = attach_accuracy(
        pd.DataFrame([{"layer": "00"}, {"layer": "01"}]),
        result,
        maximum_accuracy_drop=1.5,
    )

    assert joined.accuracy.tolist() == [85.0, 85.0]
    assert joined.accuracy_delta.tolist() == [-1.0, -1.0]
    assert joined.accuracy_constraint.tolist() == [True, True]
    assert joined.accuracy_status.eq("MODELED_WHOLE_NETWORK").all()


def test_onnsim_adapter_builds_mapping_and_parses_result(tmp_path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = _resnet_manifest(project_root)
    base_policy = _resnet_policy(project_root, manifest)
    mixed_bits = dict(base_policy.slice_bits)
    mixed_bits["00"] = 8
    policy = LayerPolicy(
        manifest=manifest,
        stationarity=base_policy.stationarity,
        slice_bits=mixed_bits,
        name=base_policy.name,
    )
    onnsim_root = tmp_path / "onnsim"
    (onnsim_root / "src/variation").mkdir(parents=True)
    (onnsim_root / "src/components").mkdir(parents=True)
    (onnsim_root / "src/variation/variation_aware_mrr.py").touch()
    (onnsim_root / "src/components/mrr.py").touch()
    checkpoint = tmp_path / "model.pth"
    model_config = tmp_path / "model.yaml"
    checkpoint.write_bytes(b"checkpoint")
    model_config.write_text("model: {}")
    captured = {}

    def fake_run(command, **kwargs):
        captured["cwd"] = kwargs["cwd"]
        mapping_index = command.index("--mapping") + 1
        captured["mapping"] = yaml.safe_load(Path(command[mapping_index]).read_text())
        captured["seeds"] = command[command.index("--seeds") + 1]
        thermal_path = Path(command[command.index("--thermal-stds") + 1])
        captured["thermal_stds"] = yaml.safe_load(thermal_path.read_text())
        captured["dac_std"] = float(command[command.index("--dac-std") + 1])
        result_path = Path(command[command.index("--output") + 1])
        result_path.write_text(json.dumps({
            "Baseline": {"accuracies": [86.0, 86.0], "losses": [0.4, 0.4]},
            "MRR Hybrid 1-bit per cycle": {
                "accuracies": [84.0, 85.0], "losses": [0.6, 0.5]
            },
            "metadata": {"seed_status": "ENFORCED_BY_OPTICALLOOP_ADAPTER"},
        }))
        return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("subprocess.run", fake_run)
    experiment = AccuracyExperimentConfig(
        network="resnet18",
        dataset="cifar10",
        checkpoint=checkpoint,
        model_config=model_config,
        runs=2,
        seeds=(7, 8),
        variation=MRRVariationConfig(thermal_std=0.05, dac_std=0.02),
    )

    result = ONNSimAccuracyBackend(onnsim_root).run(experiment, policy)

    assert captured["mapping"]["conv1"] is True
    assert captured["mapping"]["layer1.0.conv2"] is False
    assert captured["seeds"] == "7,8"
    expected_1bit_std = 0.05 * ((2**1 - 1) / (2**8 - 1)) ** 0.5
    assert set(captured["thermal_stds"]) == set(captured["mapping"])
    assert captured["thermal_stds"]["conv1"] == 0.05
    assert captured["thermal_stds"]["layer1.0.conv1"] == pytest.approx(
        expected_1bit_std
    )
    assert captured["dac_std"] == 0.02
    assert captured["cwd"] == onnsim_root
    assert result.accuracy_mean == 84.5
    assert result.accuracy_delta == -1.5
    assert result.metadata["seed_status"] == "ENFORCED_BY_OPTICALLOOP_ADAPTER"
    assert result.metadata["thermal_reference_bits"] == 8
    assert result.metadata["thermal_scaling_exponent"] == 0.5
    assert result.metadata["effective_thermal_std_by_slice_bits"]["8"] == 0.05


def test_onnsim_runner_resolves_layer_specific_thermal_std() -> None:
    pytest.importorskip("torch")

    from opticalloop.accuracy.adapters.onnsim_runner import thermal_std_for_layer

    values = {"conv1": 0.05, "layer1.0.conv1": 0.003}
    assert thermal_std_for_layer(values, "conv1") == 0.05
    assert thermal_std_for_layer(values, "layer1.0.conv1") == 0.003
    with pytest.raises(ValueError, match="Missing effective thermal std"):
        thermal_std_for_layer(values, "classifier.1")
