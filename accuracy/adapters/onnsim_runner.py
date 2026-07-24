"""Standalone ResNet18 runner using OpticalLoop's localized accuracy runtime.

The subprocess keeps PyTorch and GPU state isolated from Timeloop. Model, data
preprocessing, checkpoint compatibility, and MRR variation code are local and
do not import PlaNetZoo or ONNSim's global ``src`` package.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml


def thermal_std_for_layer(thermal_stds: dict[str, float], layer_name: str) -> float:
    """Resolve a layer-specific value at the external ONNSim boundary."""
    if layer_name not in thermal_stds:
        raise ValueError(f"Missing effective thermal std for layer: {layer_name}")
    return thermal_stds[layer_name]


SCENARIO_LABELS = {
    "baseline": "Baseline",
    "normal": "MRR Normal",
    "input": "MRR Input 1-bit per cycle",
    "input_1bit": "MRR Input 1-bit per cycle",
    "hybrid": "MRR Hybrid 1-bit per cycle",
    "hybrid_1bit": "MRR Hybrid 1-bit per cycle",
}


class ResNet18ExperimentRunner:
    """Evaluate baseline and one ONNSim variation scenario reproducibly."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def run(self) -> dict[str, object]:
        # Python seeds sys.path with this runner's directory, not subprocess
        # cwd. The adapter deliberately sets cwd to the selected ONNSim root.
        onnsim_root = str(Path.cwd())
        if onnsim_root not in sys.path:
            sys.path.insert(0, onnsim_root)
        local_accuracy_root = str(Path(__file__).resolve().parents[1])
        if local_accuracy_root not in sys.path:
            sys.path.insert(0, local_accuracy_root)
        from local_resnet import (
            Cifar10Loader,
            LocalModelConfig,
            apply_mrr_input_noise,
            apply_mrr_weight_variations,
            load_checkpoint,
        )
        from local_mrr import VariationAwareMRR

        raw_config = yaml.safe_load(self.args.config.read_text()) or {}
        model_values = raw_config.get("model", {})
        model_config = LocalModelConfig(**model_values)
        data_dir = Path(raw_config["data_dir"])
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        if str(raw_config["dataset"]).lower() != "cifar10":
            raise ValueError("local accuracy runtime supports only CIFAR-10")
        test_loader = Cifar10Loader.test_loader(
            data_dir,
            batch_size=int(raw_config["batch_size"]),
            num_workers=int(raw_config["num_workers"]),
        )
        mapping = {
            str(name): bool(value)
            for name, value in (yaml.safe_load(self.args.mapping.read_text()) or {}).items()
        }
        thermal_stds = {
            str(name): float(value)
            for name, value in (
                yaml.safe_load(self.args.thermal_stds.read_text()) or {}
            ).items()
        }
        checkpoint = torch.load(
            self.args.checkpoint, map_location=self.device, weights_only=False
        )

        def new_model():
            model = model_config.build().to(self.device)
            state = dict(checkpoint["model_state_dict"])
            return load_checkpoint(model, state).to(self.device)

        def devices(model, layer_names):
            selected = set(layer_names)
            result = {}
            for name, module in model.named_modules():
                if name not in selected or not (
                    hasattr(module, "weight_scale") and hasattr(module, "weight")
                ):
                    continue
                device = VariationAwareMRR()
                device.define_thermal_diffusion_behavior(
                    thermal_std_for_layer(thermal_stds, name)
                )
                device.define_DAC_variation_behavior(self.args.dac_std)
                device.set_weight_mapping_range(0, 3.0, 1537, -1.0, 1.0)
                result[name] = device
            return result

        baseline = {"losses": [], "accuracies": []}
        selected = {"losses": [], "accuracies": []}
        scenario = self.args.scenario
        for seed in self.args.seeds:
            self._seed(seed)
            model = new_model()
            loss, accuracy = self._evaluate(model, test_loader)
            baseline["losses"].append(loss)
            baseline["accuracies"].append(accuracy)

            if scenario == "baseline":
                selected["losses"].append(loss)
                selected["accuracies"].append(accuracy)
                continue

            self._seed(seed)
            model = new_model()
            quantized_layers = [
                name
                for name, module in model.named_modules()
                if hasattr(module, "weight_scale") and hasattr(module, "weight")
            ]
            if scenario == "normal":
                weight_devices = devices(model, quantized_layers)
                input_devices = devices(model, quantized_layers)
                apply_mrr_weight_variations(model, weight_devices, quantized_layers)
                apply_mrr_input_noise(model, input_devices, quantized_layers)
            elif scenario in {"input", "input_1bit"}:
                weight_devices = devices(model, quantized_layers)
                apply_mrr_weight_variations(model, weight_devices, quantized_layers)
            elif scenario in {"hybrid", "hybrid_1bit"}:
                input_layers = [name for name, use_input in mapping.items() if use_input]
                weight_layers = [name for name, use_input in mapping.items() if not use_input]
                apply_mrr_weight_variations(
                    model, devices(model, weight_layers), weight_layers
                )
                apply_mrr_input_noise(model, devices(model, input_layers), input_layers)
            else:
                raise ValueError(f"Unsupported ONNSim scenario: {scenario}")
            loss, accuracy = self._evaluate(model, test_loader)
            selected["losses"].append(loss)
            selected["accuracies"].append(accuracy)

        label = SCENARIO_LABELS[scenario]
        return {
            "Baseline": baseline,
            label: selected,
            "metadata": {
                "seeds": self.args.seeds,
                "seed_status": "ENFORCED_BY_OPTICALLOOP_ADAPTER",
                "device": str(self.device),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_val_acc": checkpoint.get("val_acc"),
            },
        }

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _evaluate(self, model, loader) -> tuple[float, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                loss_sum += criterion(output, target).item()
                correct += output.argmax(1).eq(target).sum().item()
                total += target.size(0)
        return loss_sum / len(loader), 100.0 * correct / total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_LABELS), required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--thermal-stds", type=Path, required=True)
    parser.add_argument("--dac-std", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.seeds = tuple(int(value) for value in args.seeds.split(","))
    return args


def main() -> None:
    args = _parse_args()
    payload = ResNet18ExperimentRunner(args).run()
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
