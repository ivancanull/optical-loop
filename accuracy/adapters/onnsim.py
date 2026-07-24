"""Subprocess adapter for the external ONNSim research code."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from opticalloop.accuracy.backend import AccuracyBackend
from opticalloop.accuracy.config import AccuracyExperimentConfig
from opticalloop.accuracy.layer_manifest import LayerPolicy
from opticalloop.accuracy.result import AccuracyResult


ONNSIM_SCENARIOS = {
    "baseline": "Baseline",
    "normal": "MRR Normal",
    "input": "MRR Input 1-bit per cycle",
    "input_1bit": "MRR Input 1-bit per cycle",
    "hybrid": "MRR Hybrid 1-bit per cycle",
    "hybrid_1bit": "MRR Hybrid 1-bit per cycle",
}


class ONNSimAccuracyBackend(AccuracyBackend):
    """Run ONNSim without importing its global `src` package into OpticalLoop."""

    def __init__(self, root: Path, *, python: str = sys.executable) -> None:
        self.root = Path(root).resolve()
        self.python = python

    def run(
        self,
        experiment: AccuracyExperimentConfig,
        policy: LayerPolicy,
    ) -> AccuracyResult:
        self._validate_inputs(experiment, policy)
        with tempfile.TemporaryDirectory(prefix="opticalloop-onnsim-") as temporary_dir:
            mapping_path = Path(temporary_dir) / "hybrid_mapping.yaml"
            thermal_stds_path = Path(temporary_dir) / "thermal_stds.yaml"
            result_path = Path(temporary_dir) / "result.json"
            mapping_path.write_text(yaml.safe_dump(policy.onnsim_mapping(), sort_keys=True))
            effective_stds = {
                module: experiment.variation.effective_thermal_std(bits)
                for module, bits in policy.onnsim_slice_bits().items()
            }
            thermal_stds_path.write_text(yaml.safe_dump(effective_stds, sort_keys=True))
            command = [
                self.python, str(Path(__file__).with_name("onnsim_runner.py")),
                "--config", str(experiment.model_config.resolve()),
                "--checkpoint", str(experiment.checkpoint.resolve()),
                "--mapping", str(mapping_path),
                "--scenario", policy.name,
                "--seeds", ",".join(str(seed) for seed in experiment.seeds),
                "--thermal-stds", str(thermal_stds_path),
                "--dac-std", str(experiment.variation.dac_std),
                "--output", str(result_path),
            ]
            completed = subprocess.run(
                command, cwd=self.root,
                capture_output=True, text=True, check=False,
            )
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"ONNSim failed with exit code {completed.returncode}: {detail}"
                )
            if not result_path.exists():
                raise FileNotFoundError(
                    f"ONNSim runner did not produce expected result: {result_path}"
                )
            payload = json.loads(result_path.read_text())
        scenario_key = ONNSIM_SCENARIOS.get(policy.name, ONNSIM_SCENARIOS["hybrid"])
        if scenario_key not in payload or "Baseline" not in payload:
            raise ValueError(f"ONNSim result is missing {scenario_key!r} or baseline")
        selected = payload[scenario_key]
        baseline = payload["Baseline"]["accuracies"]
        return AccuracyResult(
            network=experiment.network,
            scenario=policy.name,
            accuracies=tuple(float(value) for value in selected["accuracies"]),
            losses=tuple(float(value) for value in selected["losses"]),
            seeds=experiment.seeds,
            baseline_accuracy=sum(float(value) for value in baseline) / len(baseline),
            source="onnsim",
            metadata={
                "onnsim_root": str(self.root),
                "thermal_std": experiment.variation.thermal_std,
                "thermal_reference_bits": experiment.variation.thermal_reference_bits,
                "thermal_scaling_exponent": experiment.variation.thermal_scaling_exponent,
                "effective_thermal_std_by_slice_bits": {
                    str(bits): experiment.variation.effective_thermal_std(bits)
                    for bits in (1, 2, 4, 8)
                },
                "dac_std": experiment.variation.dac_std,
                "seed_status": payload.get("metadata", {}).get(
                    "seed_status", "ENFORCED_BY_OPTICALLOOP_ADAPTER"
                ),
            },
        )

    def _validate_inputs(
        self, experiment: AccuracyExperimentConfig, policy: LayerPolicy
    ) -> None:
        if experiment.network != policy.manifest.network:
            raise ValueError("experiment and layer manifest networks do not match")
        if experiment.dataset != policy.manifest.dataset:
            raise ValueError("experiment and layer manifest datasets do not match")
        required = (experiment.checkpoint, experiment.model_config)
        missing = [str(path) for path in required if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing ONNSim inputs: {missing}")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Accuracy runtime root is missing: {self.root}")
