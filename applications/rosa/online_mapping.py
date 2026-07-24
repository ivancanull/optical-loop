"""Simple online WS/IS mapping optimization using Timeloop and ONNSim."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import ClassVar, Mapping, Sequence

import yaml

from opticalloop.accuracy.adapters import ONNSimAccuracyBackend
from opticalloop.accuracy.config import AccuracyExperimentConfig, MRRVariationConfig
from opticalloop.accuracy.layer_manifest import LayerManifest, LayerPolicy
from opticalloop.backend import TimeloopBackend, TimeloopRun
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef


@dataclass(frozen=True)
class CacheProvenance:
    """Bind resumable online-mapping artifacts to their research inputs."""

    fingerprint: str
    payload: Mapping[str, object]

    SCHEMA_VERSION: ClassVar[int] = 1
    FILE_NAME: ClassVar[str] = "provenance.json"
    CACHED_ARTIFACTS: ClassVar[tuple[str, ...]] = (
        "edp_lookup.json",
        "policy_cache.json",
        "trials.csv",
        "best_policy.yaml",
        "best_result.json",
    )

    @classmethod
    def from_config(cls, config: "OptimizationConfig") -> "CacheProvenance":
        repo_root = Path(__file__).resolve().parents[2]
        input_paths = {
            "layer_manifest": config.layer_manifest,
            "initial_policy": config.initial_policy,
            "model_config": config.model_config,
            "checkpoint": config.checkpoint,
        }
        source_paths = {
            "online_mapping": Path(__file__).resolve(),
            "accuracy_runtime": repo_root / "accuracy",
            "timeloop_backend": repo_root / "backend.py",
            "simulation_result": repo_root / "result.py",
            "repository_config": repo_root / "config",
            "performance_models": repo_root / "workspace/models",
            "performance_helpers": repo_root / "workspace/scripts",
        }
        payload = {
            "schema_version": cls.SCHEMA_VERSION,
            "search": {
                "trials": config.trials,
                "optimizer_seed": config.optimizer_seed,
                "screening_seed": config.screening_seed,
                "confirmation_seeds": list(config.confirmation_seeds),
                "edp_weight": config.edp_weight,
                "accuracy_weight": config.accuracy_weight,
                "minimum_accuracy": config.minimum_accuracy,
            },
            "architecture": {
                "tiles": config.tiles,
                "pes": config.pes,
                "cols": config.cols,
                "rows": config.rows,
                "slice_bits": config.slice_bits,
            },
            "variation": {
                "thermal_std": config.thermal_std,
                "dac_std": config.dac_std,
                "thermal_scaling_exponent": config.thermal_scaling_exponent,
                "thermal_reference_bits": config.thermal_reference_bits,
            },
            "inputs": {
                name: {
                    "path": str(Path(path).resolve()),
                    "sha256": cls._digest_path(Path(path)),
                }
                for name, path in input_paths.items()
            },
            "accuracy_runtime": cls._accuracy_runtime_identity(config),
            "source_sha256": {
                # Timeloop consumes templates, component tables, and other model
                # inputs directly, so every model file participates in cache
                # compatibility. The remaining trees contain executable source.
                name: cls._digest_path(
                    path, source_files_only=name != "performance_models"
                )
                for name, path in source_paths.items()
            },
            "tools": {
                "python": platform.python_version(),
                "timeloopfe": cls._package_version("timeloopfe"),
                "accelergy": cls._package_version("accelergy"),
                "pandas": cls._package_version("pandas"),
                "pyyaml": cls._package_version("PyYAML"),
                "numpy": cls._package_version("numpy"),
                "torch": cls._package_version("torch"),
                "torchvision": cls._package_version("torchvision"),
                "nvidia_cuda_runtime": cls._package_version(
                    "nvidia-cuda-runtime-cu12"
                ),
                "nvidia_cudnn": cls._package_version("nvidia-cudnn-cu12"),
                "cuda_version": os.environ.get("CUDA_VERSION", "UNAVAILABLE"),
                "native_timeloop": cls._native_timeloop_identity(),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return cls(
            fingerprint=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            payload=payload,
        )

    def ensure_compatible(self, output_dir: Path) -> None:
        """Write provenance for a new run or reject incompatible cached data."""
        path = output_dir / self.FILE_NAME
        if path.exists():
            recorded = json.loads(path.read_text())
            if recorded.get("fingerprint") != self.fingerprint:
                raise ValueError(
                    "online-mapping cache provenance differs from the current "
                    "configuration; choose a new output directory"
                )
            return

        legacy = [name for name in self.CACHED_ARTIFACTS if (output_dir / name).exists()]
        if legacy:
            raise ValueError(
                "online-mapping cache artifacts lack provenance: "
                f"{', '.join(legacy)}; choose a new output directory"
            )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(
            {"fingerprint": self.fingerprint, "inputs": self.payload},
            indent=2,
            sort_keys=True,
        ))
        temporary.replace(path)

    @staticmethod
    def _digest_path(path: Path, *, source_files_only: bool = False) -> str:
        path = Path(path).resolve()
        if not path.exists():
            return "MISSING"
        digest = hashlib.sha256()
        files = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file()
            and "__pycache__" not in candidate.parts
            and (
                not source_files_only
                or candidate.suffix in {".py", ".yaml", ".yml"}
            )
        )
        for candidate in files:
            label = candidate.name if path.is_file() else candidate.relative_to(path).as_posix()
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            with candidate.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _accuracy_runtime_identity(
        cls, config: "OptimizationConfig"
    ) -> dict[str, str]:
        root = config.onnsim_root.resolve()
        model_config = config.model_config.resolve()
        dataset_path: Path | None = None
        if model_config.exists():
            raw = yaml.safe_load(model_config.read_text()) or {}
            if raw.get("data_dir"):
                dataset_path = Path(raw["data_dir"])
                if not dataset_path.is_absolute():
                    dataset_path = root / dataset_path
        return {
            "root": str(root),
            "dataset_path": str(dataset_path.resolve()) if dataset_path else "UNAVAILABLE",
            "dataset_sha256": (
                cls._digest_path(dataset_path) if dataset_path else "MISSING"
            ),
        }

    @classmethod
    def _native_timeloop_identity(cls) -> dict[str, object]:
        mapper_text = shutil.which("timeloop-mapper")
        if mapper_text is None:
            return {
                "mapper_path": "UNAVAILABLE",
                "mapper_sha256": "MISSING",
                "shared_libraries": {},
            }
        mapper = Path(mapper_text).resolve()
        lib_dir = mapper.parents[1] / "lib"
        libraries = {}
        if lib_dir.is_dir():
            libraries = {
                library.name: cls._digest_path(library)
                for library in sorted(lib_dir.glob("libtimeloop*.so*"))
                if library.is_file()
            }
        return {
            "mapper_path": str(mapper),
            "mapper_sha256": cls._digest_path(mapper),
            "shared_libraries": libraries,
        }

    @staticmethod
    def _package_version(distribution: str) -> str:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "UNAVAILABLE"


@dataclass(frozen=True)
class OptimizationConfig:
    """Validated inputs for one fixed-architecture online mapping search."""

    layer_manifest: Path
    initial_policy: Path
    model_config: Path
    checkpoint: Path
    onnsim_root: Path
    trials: int = 20
    optimizer_seed: int = 0
    screening_seed: int = 0
    confirmation_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    edp_weight: float = 0.5
    accuracy_weight: float = 0.5
    minimum_accuracy: float = 79.0
    thermal_std: float = 0.05
    dac_std: float = 0.02
    thermal_scaling_exponent: float = 0.5
    thermal_reference_bits: int = 8
    tiles: int = 1
    pes: int = 16
    cols: int = 8
    rows: int = 8
    slice_bits: int = 1

    def __post_init__(self) -> None:
        if self.trials < 3:
            raise ValueError("trials must include predefined, all-WS, and all-IS warm starts")
        if self.slice_bits != 1:
            raise ValueError("online mapping v1 supports only 1-bit slicing")
        if any(value <= 0 for value in (self.tiles, self.pes, self.cols, self.rows)):
            raise ValueError("architecture dimensions must be positive")
        if self.edp_weight < 0 or self.accuracy_weight < 0:
            raise ValueError("objective weights must be non-negative")
        if not math.isclose(
            self.edp_weight + self.accuracy_weight, 1.0, abs_tol=1e-12
        ):
            raise ValueError("edp_weight and accuracy_weight must sum to 1")
        if not 0 <= self.minimum_accuracy <= 100:
            raise ValueError("minimum_accuracy must be a percentage in [0, 100]")
        if not self.confirmation_seeds:
            raise ValueError("confirmation_seeds cannot be empty")
        if len(set(self.confirmation_seeds)) != len(self.confirmation_seeds):
            raise ValueError("confirmation_seeds must be unique")
        MRRVariationConfig(
            thermal_std=self.thermal_std,
            dac_std=self.dac_std,
            thermal_scaling_exponent=self.thermal_scaling_exponent,
            thermal_reference_bits=self.thermal_reference_bits,
        )

    @classmethod
    def load(cls, path: Path, *, repo_root: Path | None = None) -> "OptimizationConfig":
        path = Path(path).resolve()
        root = Path(repo_root or path.parents[2]).resolve()
        raw = yaml.safe_load(path.read_text()) or {}
        architecture = raw.get("architecture", {})

        def resolve(name: str) -> Path:
            value = Path(raw[name])
            return value if value.is_absolute() else root / value

        return cls(
            layer_manifest=resolve("layer_manifest"),
            initial_policy=resolve("initial_policy"),
            model_config=resolve("model_config"),
            checkpoint=resolve("checkpoint"),
            onnsim_root=resolve("onnsim_root"),
            trials=int(raw.get("trials", 20)),
            optimizer_seed=int(raw.get("optimizer_seed", 0)),
            screening_seed=int(raw.get("screening_seed", 0)),
            confirmation_seeds=tuple(
                int(seed) for seed in raw.get("confirmation_seeds", (0, 1, 2, 3, 4))
            ),
            edp_weight=float(raw.get("edp_weight", 0.5)),
            accuracy_weight=float(raw.get("accuracy_weight", 0.5)),
            minimum_accuracy=float(raw.get("minimum_accuracy", 79.0)),
            thermal_std=float(raw.get("thermal_std", 0.05)),
            dac_std=float(raw.get("dac_std", 0.02)),
            thermal_scaling_exponent=float(raw.get("thermal_scaling_exponent", 0.5)),
            thermal_reference_bits=int(raw.get("thermal_reference_bits", 8)),
            tiles=int(architecture.get("tiles", 1)),
            pes=int(architecture.get("pes", 16)),
            cols=int(architecture.get("cols", 8)),
            rows=int(architecture.get("rows", 8)),
            slice_bits=int(raw.get("slice_bits", 1)),
        )

    def score(
        self,
        *,
        edp_j_s: float,
        accuracy: float,
        reference_edp_j_s: float,
        baseline_accuracy: float,
    ) -> tuple[float | None, bool]:
        """Return normalized weighted score and hard-floor feasibility."""
        if edp_j_s <= 0 or reference_edp_j_s <= 0:
            raise ValueError("EDP values must be positive")
        if baseline_accuracy <= self.minimum_accuracy:
            raise ValueError("baseline accuracy must exceed minimum_accuracy")
        if accuracy < self.minimum_accuracy:
            return None, False
        edp_cost = edp_j_s / reference_edp_j_s
        accuracy_cost = (baseline_accuracy - accuracy) / (
            baseline_accuracy - self.minimum_accuracy
        )
        return (
            self.edp_weight * edp_cost + self.accuracy_weight * accuracy_cost,
            True,
        )


class PolicyEvaluator:
    """Evaluate and persist Timeloop EDP plus whole-network ONNSim accuracy."""

    def __init__(
        self,
        config: OptimizationConfig,
        output_dir: Path,
        *,
        timeloop_backend=None,
        accuracy_backend=None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = CacheProvenance.from_config(config)
        self.provenance.ensure_compatible(self.output_dir)
        self.manifest = LayerManifest.load(config.layer_manifest)
        if len(self.manifest.layers) != 21:
            raise ValueError(
                f"ResNet18 online mapping requires 21 layers, found {len(self.manifest.layers)}"
            )
        self.timeloop_backend = timeloop_backend or TimeloopBackend()
        self.accuracy_backend = accuracy_backend or ONNSimAccuracyBackend(
            config.onnsim_root
        )
        self.edp_path = self.output_dir / "edp_lookup.json"
        self.policy_cache_path = self.output_dir / "policy_cache.json"
        self.edp_lookup = self._read_json(self.edp_path)
        self.policy_cache = self._read_json(self.policy_cache_path)

    def precompute_edp(self) -> None:
        pending = []
        for layer in self.manifest.layers:
            for stationarity in ("WS", "IS"):
                key = self._edp_key(layer.layer_id, stationarity)
                if key in self.edp_lookup:
                    continue
                architecture = self._architecture(stationarity)
                run = TimeloopRun(
                    layer=TimeloopLayerRef(
                        network=self.manifest.network, layer_path=layer.workload
                    ),
                    architecture=architecture,
                )
                pending.append((key, layer.layer_id, stationarity, architecture, run))

        if not pending:
            return
        results = self.timeloop_backend.run_batch(
            [item[-1] for item in pending], n_jobs=min(4, len(pending))
        )
        for (key, layer_id, stationarity, architecture, _), result in zip(
            pending, results
        ):
            self.edp_lookup[key] = {
                "layer": layer_id,
                "stationarity": stationarity,
                "macro": architecture.macro,
                "energy_j": result.energy_j,
                "latency_s": result.latency_s,
                "cycles": result.cycles,
                "source": result.source,
            }
            self._write_json(self.edp_path, self.edp_lookup)

    def evaluate(
        self, stationarity: Mapping[str, str], seeds: Sequence[int]
    ) -> dict[str, object]:
        policy = self.make_policy(stationarity)
        policy_key = self.policy_key(stationarity)
        seed_key = ",".join(str(seed) for seed in seeds)
        cached = self.policy_cache.get(policy_key, {}).get(seed_key)
        if cached is not None:
            return dict(cached)

        self.precompute_edp()
        energy = sum(
            float(self.edp_lookup[self._edp_key(layer.layer_id, stationarity[layer.layer_id])]["energy_j"])
            for layer in self.manifest.layers
        )
        latency = sum(
            float(self.edp_lookup[self._edp_key(layer.layer_id, stationarity[layer.layer_id])]["latency_s"])
            for layer in self.manifest.layers
        )
        experiment = AccuracyExperimentConfig(
            network=self.manifest.network,
            dataset=self.manifest.dataset,
            checkpoint=self.config.checkpoint,
            model_config=self.config.model_config,
            runs=len(seeds),
            seeds=tuple(int(seed) for seed in seeds),
            variation=MRRVariationConfig(
                thermal_std=self.config.thermal_std,
                dac_std=self.config.dac_std,
                thermal_scaling_exponent=self.config.thermal_scaling_exponent,
                thermal_reference_bits=self.config.thermal_reference_bits,
            ),
        )
        accuracy = self.accuracy_backend.run(experiment, policy)
        row = {
            "policy_key": policy_key,
            "stationarity": dict(stationarity),
            "seeds": list(seeds),
            "energy_j": energy,
            "latency_s": latency,
            "edp_j_s": energy * latency,
            "accuracy": accuracy.accuracy_mean,
            "accuracy_std": accuracy.accuracy_std,
            "baseline_accuracy": accuracy.baseline_accuracy,
            "accuracy_source": accuracy.source,
        }
        self.policy_cache.setdefault(policy_key, {})[seed_key] = row
        self._write_json(self.policy_cache_path, self.policy_cache)
        return dict(row)

    def make_policy(self, stationarity: Mapping[str, str]) -> LayerPolicy:
        self._validate_stationarity(stationarity)
        return LayerPolicy(
            manifest=self.manifest,
            stationarity=dict(stationarity),
            slice_bits={
                layer.layer_id: self.config.slice_bits for layer in self.manifest.layers
            },
            name="hybrid",
        )

    def policy_key(self, stationarity: Mapping[str, str]) -> str:
        self._validate_stationarity(stationarity)
        return "".join(
            "1" if stationarity[layer.layer_id] == "IS" else "0"
            for layer in self.manifest.layers
        )

    def _validate_stationarity(self, stationarity: Mapping[str, str]) -> None:
        expected = {layer.layer_id for layer in self.manifest.layers}
        if set(stationarity) != expected:
            raise ValueError("candidate must specify every canonical layer exactly once")
        invalid = set(stationarity.values()) - {"WS", "IS"}
        if invalid:
            raise ValueError(f"unsupported stationarity choices: {sorted(invalid)}")

    def _architecture(self, stationarity: str) -> MRRMacroConfig:
        return MRRMacroConfig(
            n_tiles=self.config.tiles,
            n_pes=self.config.pes,
            n_cols=self.config.cols,
            n_rows=self.config.rows,
            macro="mrr_ws_osa" if stationarity == "WS" else "mrr_is_osa",
            front_mrr_slice_bits=self.config.slice_bits,
            max_utilization=False,
        )

    @staticmethod
    def _edp_key(layer_id: str, stationarity: str) -> str:
        return f"{layer_id}:{stationarity}"

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)


class MappingOptimizer:
    """Deterministic hill climbing over one WS/IS decision per layer."""

    TRIAL_FIELDS = (
        "trial",
        "kind",
        "policy_key",
        "changed_layers",
        "energy_j",
        "latency_s",
        "edp_j_s",
        "accuracy",
        "accuracy_std",
        "baseline_accuracy",
        "score",
        "feasible",
        "accepted",
    )

    def __init__(
        self,
        config: OptimizationConfig,
        evaluator: PolicyEvaluator,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.output_dir = evaluator.output_dir
        self.rng = random.Random(config.optimizer_seed)

    def optimize(self) -> dict[str, object]:
        reference_policy = LayerPolicy.load(
            self.config.initial_policy, self.evaluator.manifest
        )
        reference = dict(reference_policy.stationarity)
        screening_seeds = (self.config.screening_seed,)
        trial_rows: list[dict[str, object]] = []
        seen: set[str] = set()

        reference_result = self.evaluator.evaluate(reference, screening_seeds)
        reference_edp = float(reference_result["edp_j_s"])
        baseline = float(reference_result["baseline_accuracy"])
        current = reference
        current_score, current_feasible = self.config.score(
            edp_j_s=reference_edp,
            accuracy=float(reference_result["accuracy"]),
            reference_edp_j_s=reference_edp,
            baseline_accuracy=baseline,
        )
        self._record(
            trial_rows,
            seen,
            0,
            "predefined",
            reference,
            (),
            reference_result,
            current_score,
            current_feasible,
            current_feasible,
        )

        warm_starts = (
            ("all_ws", {layer.layer_id: "WS" for layer in self.evaluator.manifest.layers}),
            ("all_is", {layer.layer_id: "IS" for layer in self.evaluator.manifest.layers}),
        )
        for kind, candidate in warm_starts:
            result = self.evaluator.evaluate(candidate, screening_seeds)
            score, feasible = self._score(result, reference_edp, baseline)
            accepted = self._is_better(score, feasible, current_score, current_feasible)
            if accepted:
                current, current_score, current_feasible = candidate, score, feasible
            self._record(
                trial_rows,
                seen,
                len(trial_rows),
                kind,
                candidate,
                (),
                result,
                score,
                feasible,
                accepted,
            )

        while len(trial_rows) < self.config.trials:
            candidate, changed = self._unique_mutation(current, seen)
            result = self.evaluator.evaluate(candidate, screening_seeds)
            score, feasible = self._score(result, reference_edp, baseline)
            accepted = self._is_better(score, feasible, current_score, current_feasible)
            if accepted:
                current, current_score, current_feasible = candidate, score, feasible
            self._record(
                trial_rows,
                seen,
                len(trial_rows),
                "mutation",
                candidate,
                changed,
                result,
                score,
                feasible,
                accepted,
            )

        if not current_feasible:
            payload = {
                "status": "no_feasible_policy",
                "phase": "screening",
                "reference_policy_key": self.evaluator.policy_key(reference),
                "trials": len(trial_rows),
            }
            self._clear_best_policy()
            self._write_best_result(payload)
            return payload

        winner_confirmation = self.evaluator.evaluate(
            current, self.config.confirmation_seeds
        )
        reference_confirmation = self.evaluator.evaluate(
            reference, self.config.confirmation_seeds
        )
        confirmed_baseline = float(reference_confirmation["baseline_accuracy"])
        winner_score, winner_feasible = self._score(
            winner_confirmation, reference_edp, confirmed_baseline
        )
        reference_score, reference_feasible = self._score(
            reference_confirmation, reference_edp, confirmed_baseline
        )
        if not self._is_better(
            winner_score, winner_feasible, reference_score, reference_feasible
        ):
            current = reference
            winner_confirmation = reference_confirmation
            winner_score = reference_score
            winner_feasible = reference_feasible

        if not winner_feasible:
            payload = {
                "status": "no_feasible_policy",
                "phase": "confirmation",
                "reference_policy_key": self.evaluator.policy_key(reference),
                "trials": len(trial_rows),
            }
            self._clear_best_policy()
            self._write_best_result(payload)
            return payload

        self._write_policy(current)
        payload = {
            "status": "success",
            "trials": len(trial_rows),
            "selected_policy_key": self.evaluator.policy_key(current),
            "selected_score": winner_score,
            "selected_feasible": winner_feasible,
            "selected_result": winner_confirmation,
            "reference_policy_key": self.evaluator.policy_key(reference),
            "reference_score": reference_score,
            "reference_result": reference_confirmation,
        }
        self._write_best_result(payload)
        return payload

    def _unique_mutation(
        self, current: Mapping[str, str], seen: set[str]
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        layer_ids = [layer.layer_id for layer in self.evaluator.manifest.layers]
        for _ in range(1_000):
            count = 1 if self.rng.random() < 0.8 else 2
            changed = tuple(sorted(self.rng.sample(layer_ids, count)))
            candidate = dict(current)
            for layer_id in changed:
                candidate[layer_id] = "IS" if candidate[layer_id] == "WS" else "WS"
            if self.evaluator.policy_key(candidate) not in seen:
                return candidate, changed
        raise RuntimeError("could not generate a unique mapping candidate")

    def _score(
        self, result: Mapping[str, object], reference_edp: float, baseline: float
    ) -> tuple[float | None, bool]:
        return self.config.score(
            edp_j_s=float(result["edp_j_s"]),
            accuracy=float(result["accuracy"]),
            reference_edp_j_s=reference_edp,
            baseline_accuracy=baseline,
        )

    @staticmethod
    def _is_better(
        score: float | None,
        feasible: bool,
        current_score: float | None,
        current_feasible: bool,
    ) -> bool:
        if not feasible:
            return False
        if not current_feasible or current_score is None:
            return True
        return score is not None and score < current_score

    def _record(
        self,
        rows: list[dict[str, object]],
        seen: set[str],
        trial: int,
        kind: str,
        policy: Mapping[str, str],
        changed: Sequence[str],
        result: Mapping[str, object],
        score: float | None,
        feasible: bool,
        accepted: bool,
    ) -> None:
        policy_key = self.evaluator.policy_key(policy)
        seen.add(policy_key)
        rows.append(
            {
                "trial": trial,
                "kind": kind,
                "policy_key": policy_key,
                "changed_layers": ",".join(changed),
                "energy_j": result["energy_j"],
                "latency_s": result["latency_s"],
                "edp_j_s": result["edp_j_s"],
                "accuracy": result["accuracy"],
                "accuracy_std": result["accuracy_std"],
                "baseline_accuracy": result["baseline_accuracy"],
                "score": score,
                "feasible": feasible,
                "accepted": accepted,
            }
        )
        with (self.output_dir / "trials.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=self.TRIAL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _write_policy(self, stationarity: Mapping[str, str]) -> None:
        payload = {
            "name": "hybrid",
            "layers": {
                layer.layer_id: {
                    "stationarity": stationarity[layer.layer_id],
                    "slice_bits": self.config.slice_bits,
                }
                for layer in self.evaluator.manifest.layers
            },
        }
        (self.output_dir / "best_policy.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False)
        )

    def _write_best_result(self, payload: Mapping[str, object]) -> None:
        PolicyEvaluator._write_json(self.output_dir / "best_result.json", payload)

    def _clear_best_policy(self) -> None:
        path = self.output_dir / "best_policy.yaml"
        if path.exists():
            path.unlink()


def run_online_mapping(
    config_path: Path,
    output_dir: Path,
    *,
    timeloop_backend=None,
    accuracy_backend=None,
) -> dict[str, object]:
    """Load configuration and run one resumable online mapping search."""
    config = OptimizationConfig.load(config_path)
    for required in (
        config.layer_manifest,
        config.initial_policy,
        config.model_config,
        config.checkpoint,
        config.onnsim_root,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    evaluator = PolicyEvaluator(
        config,
        output_dir,
        timeloop_backend=timeloop_backend,
        accuracy_backend=accuracy_backend,
    )
    return MappingOptimizer(config, evaluator).optimize()
