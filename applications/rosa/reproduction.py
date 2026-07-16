"""Clean-checkout DAC26 EDP experiment orchestration and verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd
import yaml

from opticalloop.backend import TimeloopBackend
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.result import SimulationResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReproductionJob:
    job_id: str
    network: str
    layer: str
    variant: str
    macro: str
    architecture: str
    tiles: int
    pes: int
    cols: int
    rows: int
    slice_bits: int
    temporal_slices: int
    radix: int


def _execute_native_job(
    job: ReproductionJob, system: str, frequency_hz: float
) -> tuple[Optional[SimulationResult], Optional[str]]:
    """Process-pool entrypoint; keep raw mapper calls inside the backend adapter."""
    try:
        result = TimeloopBackend().run_layer(
            TimeloopLayerRef(network=job.network, layer_path=job.layer),
            MRRMacroConfig(
                n_tiles=job.tiles, n_pes=job.pes, n_cols=job.cols, n_rows=job.rows,
                macro=job.macro, system=system, max_utilization=False,
                input_slice_bits=job.slice_bits,
                frequency_hz=frequency_hz,
            ),
        )
        return result, None
    except Exception as error:
        return None, repr(error)


class ExperimentManifest:
    """Validated, machine-readable definition of the paper experiment."""

    def __init__(self, path: Path, repo_root: Optional[Path] = None) -> None:
        self.path = Path(path).resolve()
        self.repo_root = (repo_root or self.path.parents[2]).resolve()
        self.raw = yaml.safe_load(self.path.read_text())
        self.validate()

    @property
    def digest(self) -> str:
        return _canonical_hash(self.raw)

    @property
    def networks(self) -> tuple[str, ...]:
        return tuple(self.raw["workloads"])

    def layers(self, network: str) -> tuple[str, ...]:
        spec = self.raw["workloads"][network]
        paths = sorted(self.repo_root.glob(spec["layer_glob"]))
        return tuple(f"{network}/{path.stem}" for path in paths)

    def architecture(self, name: str) -> Mapping[str, object]:
        matches = [a for a in self.raw["architectures"] if a["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one architecture named {name!r}")
        return matches[0]

    def jobs(
        self, tier: str, *, manifest_digest: Optional[str] = None
    ) -> tuple[ReproductionJob, ...]:
        if tier not in {"smoke", "full"}:
            raise ValueError(f"Unknown reproduction tier: {tier}")
        architectures = self.raw["architectures"]
        if tier == "smoke":
            architectures = [self.architecture(self.raw["smoke"]["architecture"])]
        jobs = []
        for network in self.networks:
            layers = self.layers(network)
            if tier == "smoke":
                layers = (f"{network}/{self.raw['smoke']['layers'][network]}",)
                smoke_path = self.manifest_path_for_layer(layers[0])
                if not smoke_path.exists():
                    raise ValueError(f"Smoke layer does not exist: {layers[0]}")
            for variant, variant_spec in self.raw["variants"].items():
                slice_bits = int(variant_spec.get("slice_bits", 1))
                if slice_bits not in {1, 2, 4, 8}:
                    raise ValueError(f"Unsupported slice width for {variant}: {slice_bits}")
                for architecture in architectures:
                    for layer in layers:
                        identity = {
                            # Validation of an existing immutable run uses the
                            # digest recorded when that run was created. This
                            # permits analysis-only settings (such as a larger
                            # exact-frontier safety cap) to evolve without
                            # relabeling native mapper checkpoints.
                            "manifest": manifest_digest or self.digest,
                            "network": network,
                            "layer": layer,
                            "variant": variant,
                            "slice_bits": slice_bits,
                            "architecture": architecture["name"],
                        }
                        jobs.append(
                            ReproductionJob(
                                job_id=_canonical_hash(identity)[:20],
                                network=network,
                                layer=layer,
                                variant=variant,
                                macro=variant_spec["macro"],
                                architecture=architecture["name"],
                                tiles=int(architecture["tiles"]),
                                pes=int(architecture["pes"]),
                                cols=int(architecture["cols"]),
                                rows=int(architecture["rows"]),
                                slice_bits=slice_bits,
                                temporal_slices=(8 + slice_bits - 1) // slice_bits,
                                radix=2 ** slice_bits,
                            )
                        )
        return tuple(jobs)

    def manifest_path_for_layer(self, layer: str) -> Path:
        return self.repo_root / "workspace/models/workloads" / f"{layer}.yaml"

    def validate(self) -> None:
        required = {"workloads", "variants", "architectures", "constraints", "tolerances"}
        missing = required - set(self.raw)
        if missing:
            raise ValueError(f"Manifest missing keys: {sorted(missing)}")
        names = [architecture["name"] for architecture in self.raw["architectures"]]
        if len(names) != len(set(names)):
            raise ValueError("Architecture names must be unique")
        constraints = self.raw["constraints"]
        for architecture in self.raw["architectures"]:
            values = [architecture[key] for key in ("tiles", "pes", "cols", "rows")]
            if any(not isinstance(value, int) or value <= 0 for value in values):
                raise ValueError(f"Invalid architecture dimensions: {architecture}")
            if architecture["candidate"]:
                if architecture["cols"] > constraints["max_candidate_cols"]:
                    raise ValueError(f"Candidate exceeds column constraint: {architecture['name']}")
                mrrs = architecture["pes"] * architecture["cols"] * architecture["rows"]
                if mrrs > constraints["max_weight_mrrs"]:
                    raise ValueError(f"Candidate exceeds MRR constraint: {architecture['name']}")
        for network, spec in self.raw["workloads"].items():
            actual = len(tuple(self.repo_root.glob(spec["layer_glob"])))
            if actual != spec["expected_layers"]:
                raise ValueError(
                    f"{network} expected {spec['expected_layers']} layers, found {actual}"
                )
        if sum(spec["expected_layers"] for spec in self.raw["workloads"].values()) != 352:
            raise ValueError("DAC26 manifest must contain exactly 352 workload layers")


class EnvironmentDoctor:
    """Read-only preflight checks for a native reproduction environment."""

    def __init__(self, manifest: ExperimentManifest) -> None:
        self.manifest = manifest

    def check(self) -> pd.DataFrame:
        rows = []
        for executable in ("timeloop-mapper", "accelergy"):
            path = shutil.which(executable)
            rows.append(self._row(f"executable:{executable}", bool(path), path or "not found"))
        mapper = shutil.which("timeloop-mapper")
        libraries_ok, libraries_detail = self._shared_libraries(mapper)
        rows.append(self._row("timeloop:shared_libraries", libraries_ok, libraries_detail))
        plugins_ok, plugins_detail = self._accelergy_plugins()
        rows.append(self._row("accelergy:plugin_discovery", plugins_ok, plugins_detail))
        rows.append(self._row("python:timeloopfe", self._importable("timeloopfe"), sys.executable))
        rows.append(self._row("python:yaml", self._importable("yaml"), sys.executable))
        rows.append(self._row("python:pandas", self._importable("pandas"), sys.executable))
        macros_root = self.manifest.repo_root / "workspace/models/arch/1_macro"
        for variant, spec in self.manifest.raw["variants"].items():
            path = macros_root / spec["macro"]
            rows.append(self._row(f"macro:{variant}", path.is_dir(), str(path)))
        output_root = self.manifest.repo_root / "reproduction-runs"
        writable = os.access(output_root.parent, os.W_OK)
        rows.append(self._row("output:writable", writable, str(output_root)))
        rows.append(self._row("manifest:352_layers", True, self.manifest.digest))
        return pd.DataFrame(rows)

    def provenance(self) -> dict[str, object]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "timeloop_mapper": self._executable_identity("timeloop-mapper"),
            "timeloopfe": self._package_version("timeloopfe"),
            "accelergy": self._package_version("accelergy"),
            "git_commit": self._git_commit(),
            "git_dirty": self._git_dirty(),
        }

    @staticmethod
    def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
        return {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}

    @staticmethod
    def _importable(module: str) -> bool:
        try:
            __import__(module)
            return True
        except ImportError:
            return False

    @staticmethod
    def _executable_identity(executable: str) -> Optional[str]:
        path = shutil.which(executable)
        if not path:
            return None
        completed = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        output = (completed.stdout or completed.stderr).strip().splitlines()
        if completed.returncode == 0 and output:
            return output[0]
        stat = Path(path).stat()
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return f"{Path(path).name} size={stat.st_size} sha256={digest}"

    @staticmethod
    def _package_version(package: str) -> Optional[str]:
        try:
            return importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _shared_libraries(mapper: Optional[str]) -> tuple[bool, str]:
        if not mapper:
            return False, "timeloop-mapper not found"
        completed = subprocess.run(
            ["ldd", mapper], capture_output=True, text=True, timeout=10, check=False
        )
        output = "\n".join((completed.stdout, completed.stderr)).strip()
        missing = [line.strip() for line in output.splitlines() if "not found" in line]
        static_binary = any(
            marker in output.lower()
            for marker in ("statically linked", "not a dynamic executable")
        )
        passed = not missing and (completed.returncode == 0 or static_binary)
        detail = "; ".join(missing) or (
            "static executable" if static_binary else "all resolved"
        )
        return passed, detail

    @staticmethod
    def _accelergy_plugins() -> tuple[bool, str]:
        executable = shutil.which("accelergy")
        if not executable:
            return False, "accelergy not found"
        completed = subprocess.run(
            [executable, "--list-components"], capture_output=True, text=True,
            timeout=60, check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        # A discovered installation prints a non-trivial component/action catalog.
        passed = completed.returncode == 0 and len(output.splitlines()) >= 5
        return passed, f"returncode={completed.returncode}, lines={len(output.splitlines())}"

    def _git_commit(self) -> Optional[str]:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.manifest.repo_root,
            capture_output=True, text=True, check=False,
        )
        return completed.stdout.strip() or None

    def _git_dirty(self) -> Optional[bool]:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.manifest.repo_root,
            capture_output=True, text=True, check=False,
        )
        return bool(completed.stdout) if completed.returncode == 0 else None


class ReproductionRunner:
    """Checkpoint each mapper job in an immutable manifest-keyed run directory."""

    def __init__(
        self,
        manifest: ExperimentManifest,
        run_root: Path,
        backend: Optional[TimeloopBackend] = None,
    ) -> None:
        self.manifest = manifest
        self.run_root = Path(run_root).resolve()
        self.backend = backend or TimeloopBackend()

    def run(
        self,
        tier: str,
        *,
        resume: bool = True,
        fail_fast: bool = False,
        workers: int = 4,
    ) -> Path:
        if workers <= 0:
            raise ValueError("workers must be positive")
        provenance = EnvironmentDoctor(self.manifest).provenance()
        run_id = _canonical_hash(
            {"manifest": self.manifest.digest, "tier": tier, "provenance": provenance}
        )[:16]
        run_dir = self.run_root / f"{self.manifest.raw['name']}-{tier}-{run_id}"
        jobs_dir = run_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "run.json"
        if not resume and metadata_path.exists():
            raise ValueError(
                "Run directories are immutable; --no-resume requires an unused --run-root"
            )
        jobs = self.manifest.jobs(tier)
        metadata = self._load_or_initialize(metadata_path, tier, jobs, provenance)
        pending = [
            job for job in jobs
            if not (
                resume
                and (jobs_dir / f"{job.job_id}.json").exists()
                and json.loads((jobs_dir / f"{job.job_id}.json").read_text()).get("status") == "success"
            )
        ]
        executor_type = ThreadPoolExecutor if self.backend is not None else ProcessPoolExecutor
        # A caller-supplied backend is primarily a test/embedding seam and may
        # not be picklable. Production native runs use processes because YAML
        # preparation is CPU-bound and TimeloopFE serializes threaded calls.
        injected_backend = self.backend
        if type(self.backend) is TimeloopBackend and self.backend.quick_run is None and self.backend.scripts_dir is None:
            executor_type = ProcessPoolExecutor
            injected_backend = None
        else:
            executor_type = ThreadPoolExecutor
        with executor_type(max_workers=workers) as executor:
            if injected_backend is None:
                futures = {
                    executor.submit(
                        _execute_native_job,
                        job,
                        self.manifest.raw["system"],
                        float(self.manifest.raw["frequency_hz"]),
                    ): job
                    for job in pending
                }
            else:
                futures = {executor.submit(self._run_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                if injected_backend is None:
                    result, error = future.result()
                    payload = (
                        self._success_payload(job, result)
                        if result is not None
                        else self._failure_payload(job, error or "unknown mapper failure")
                    )
                else:
                    payload = future.result()
                self._atomic_json(jobs_dir / f"{job.job_id}.json", payload)
                self._write_job_log(logs_dir / f"{job.job_id}.log", payload)
                if payload["status"] == "failed" and fail_fast:
                    for remaining in futures:
                        remaining.cancel()
                    break
        self._finalize(metadata_path, metadata, jobs_dir, jobs)
        return run_dir

    def _run_job(self, job: ReproductionJob) -> dict[str, object]:
        try:
            result = self.backend.run_layer(
                TimeloopLayerRef(network=job.network, layer_path=job.layer),
                MRRMacroConfig(
                    n_tiles=job.tiles, n_pes=job.pes, n_cols=job.cols, n_rows=job.rows,
                    macro=job.macro, system=self.manifest.raw["system"], max_utilization=False,
                    input_slice_bits=job.slice_bits,
                    frequency_hz=float(self.manifest.raw["frequency_hz"]),
                ),
            )
            return self._success_payload(job, result)
        except Exception as error:  # Job boundary: persist mapper failure for resume/reporting.
            return self._failure_payload(job, repr(error))

    @staticmethod
    def _failure_payload(job: ReproductionJob, error: str) -> dict[str, object]:
        return {
            "status": "failed", "job": asdict(job), "error": error,
            "finished_at": _utc_now(),
        }

    def _load_or_initialize(
        self, path: Path, tier: str, jobs: Sequence[ReproductionJob], provenance: Mapping[str, object]
    ) -> dict:
        if path.exists():
            metadata = json.loads(path.read_text())
            if metadata["manifest_digest"] != self.manifest.digest or metadata["tier"] != tier:
                raise ValueError("Refusing to mix results from a different manifest or tier")
            if metadata["provenance"] != provenance:
                raise ValueError("Refusing to mix results from a different toolchain/source provenance")
            return metadata
        metadata = {
            "schema_version": 1, "run_id": path.parent.name, "tier": tier,
            "manifest_digest": self.manifest.digest, "manifest": self.manifest.raw,
            "created_at": _utc_now(), "command": sys.argv,
            "provenance": dict(provenance),
            "expected_jobs": len(jobs), "status": "running",
        }
        self._atomic_json(path, metadata)
        return metadata

    def _finalize(self, path: Path, metadata: dict, jobs_dir: Path, jobs: Sequence[ReproductionJob]) -> None:
        payloads = [json.loads((jobs_dir / f"{job.job_id}.json").read_text()) for job in jobs if (jobs_dir / f"{job.job_id}.json").exists()]
        succeeded = sum(payload["status"] == "success" for payload in payloads)
        failed = sum(payload["status"] == "failed" for payload in payloads)
        metadata.update({
            "updated_at": _utc_now(), "successful_jobs": succeeded, "failed_jobs": failed,
            "remaining_jobs": len(jobs) - len(payloads),
            "status": "complete" if succeeded == len(jobs) else "incomplete",
        })
        self._atomic_json(path, metadata)

    @staticmethod
    def _success_payload(job: ReproductionJob, result: SimulationResult) -> dict:
        return {
            "status": "success", "job": asdict(job), "finished_at": _utc_now(),
            "metrics": {
                "cycles": result.cycles, "latency_s": result.latency_s,
                "energy_j": result.energy_j, "edp_j_s": result.energy_j * result.latency_s,
                "area_mm2": result.area_mm2, "compute": result.compute,
                "cycle_seconds": result.cycle_seconds, "tops": result.tops,
                "energy_breakdown": result.energy_breakdown,
                "area_breakdown": result.area_breakdown,
                "power_breakdown": result.power_breakdown,
            },
        }

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)

    @staticmethod
    def _write_job_log(path: Path, payload: Mapping[str, object]) -> None:
        job = payload["job"]
        lines = [
            f"job_id={job['job_id']}", f"network={job['network']}",
            f"layer={job['layer']}", f"variant={job['variant']}",
            f"architecture={job['architecture']}", f"status={payload['status']}",
            f"finished_at={payload['finished_at']}",
        ]
        if "error" in payload:
            lines.append(f"error={payload['error']}")
        path.write_text("\n".join(lines) + "\n")


class ReproductionAnalyzer:
    """Build aggregates and a reader-facing verification report from one run."""

    def __init__(self, manifest: ExperimentManifest, run_dir: Path) -> None:
        self.manifest = manifest
        self.run_dir = Path(run_dir).resolve()

    def raw_dataframe(self) -> pd.DataFrame:
        rows = []
        for path in sorted((self.run_dir / "jobs").glob("*.json")):
            payload = json.loads(path.read_text())
            if payload["status"] != "success":
                continue
            rows.append({**payload["job"], **payload["metrics"]})
        return pd.DataFrame(rows)

    def analyze(self, *, execute_notebook: bool = False) -> Mapping[str, Path]:
        artifacts = self.run_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        raw = self.raw_dataframe()
        raw.to_csv(artifacts / "layer_results.csv", index=False)
        aggregates = self._aggregate(raw)
        aggregates.to_csv(artifacts / "network_architecture_metrics.csv", index=False)
        headline = self._headline(aggregates)
        (artifacts / "headline.json").write_text(json.dumps(headline, indent=2) + "\n")
        checks = ReproductionValidator(self.manifest, self.run_dir).validate(raw, aggregates, headline)
        checks.to_csv(artifacts / "validation.csv", index=False)
        report = self._markdown_report(checks, headline)
        (artifacts / "REPORT.md").write_text(report)
        outputs = {name: artifacts / name for name in (
            "layer_results.csv", "network_architecture_metrics.csv", "headline.json",
            "validation.csv", "REPORT.md",
        )}
        if headline.get("available"):
            outputs["architecture_edp.png"] = self._write_architecture_plot(
                aggregates, artifacts / "architecture_edp.png"
            )
        if execute_notebook:
            outputs["notebook.ipynb"] = self._execute_notebook(artifacts)
        return outputs

    @staticmethod
    def _write_architecture_plot(aggregates: pd.DataFrame, path: Path) -> Path:
        import matplotlib.pyplot as plt

        rows = []
        for (architecture, variant), group in aggregates.groupby(["architecture", "variant"]):
            values = group.edp_j_s.astype(float)
            rows.append({
                "architecture": architecture, "variant": variant,
                "geometric_mean_edp": math.exp(sum(map(math.log, values)) / len(values)),
            })
        chart = pd.DataFrame(rows).pivot(
            index="architecture", columns="variant", values="geometric_mean_edp"
        )
        axis = chart.plot.bar(figsize=(11, 5), logy=True, ylabel="Geometric-mean EDP (J·s)")
        axis.set_title("DAC26 six-workload architecture sweep")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def _execute_notebook(self, artifacts: Path) -> Path:
        source = self.manifest.repo_root / "examples/rosa/dac26_edp_reproduction.ipynb"
        output = artifacts / "dac26_edp_reproduction.executed.ipynb"
        environment = os.environ.copy()
        environment["OPTICALLOOP_RUN_DIR"] = str(self.run_dir)
        kernel_root = artifacts / ".jupyter/kernels/opticalloop-reproduction"
        kernel_root.mkdir(parents=True, exist_ok=True)
        (kernel_root / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "OpticalLoop reproduction", "language": "python",
        }))
        environment["JUPYTER_PATH"] = str(artifacts / ".jupyter")
        completed = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                "--execute", str(source), "--output", output.name,
                "--output-dir", str(artifacts), "--ExecutePreprocessor.timeout=3600",
                "--ExecutePreprocessor.kernel_name=opticalloop-reproduction",
            ],
            cwd=self.manifest.repo_root, env=environment, text=True,
            capture_output=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"Notebook execution failed:\n{completed.stdout}\n{completed.stderr}")
        return output

    @staticmethod
    def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        grouped = raw.groupby(["network", "variant", "architecture", "tiles", "pes", "cols", "rows"], sort=False)
        rows = []
        for keys, group in grouped:
            energy = group["energy_j"].sum()
            latency = group["latency_s"].sum()
            rows.append(dict(zip(
                ("network", "variant", "architecture", "tiles", "pes", "cols", "rows"), keys
            ), layers=len(group), energy_j=energy, latency_s=latency, edp_j_s=energy * latency))
        return pd.DataFrame(rows)

    def _headline(self, aggregates: pd.DataFrame) -> dict[str, object]:
        metadata = json.loads((self.run_dir / "run.json").read_text())
        if metadata["tier"] != "full" or aggregates.empty:
            return {"available": False, "reason": "headline metrics require a complete full-tier run"}
        summary = []
        for (variant, architecture), group in aggregates.groupby(["variant", "architecture"]):
            values = group["edp_j_s"].astype(float)
            summary.append({"variant": variant, "architecture": architecture,
                            "geometric_mean_edp": math.exp(sum(map(math.log, values)) / len(values))})
        summary_df = pd.DataFrame(summary)
        no_osa = summary_df[summary_df["variant"] == "no_osa"]
        candidates = {a["name"] for a in self.manifest.raw["architectures"] if a["candidate"]}
        winner = no_osa[no_osa["architecture"].isin(candidates)].sort_values("geometric_mean_edp").iloc[0]
        baseline = self.manifest.raw["baselines"]
        value = lambda variant, name: float(summary_df[(summary_df.variant == variant) & (summary_df.architecture == name)].iloc[0].geometric_mean_edp)
        optimized = value("no_osa", baseline["optimized"])
        return {
            "available": True, "winner": winner["architecture"],
            "optimized_vs_compact_reduction": 1 - optimized / value("no_osa", baseline["compact"]),
            "optimized_vs_deap_reduction": 1 - optimized / value("no_osa", baseline["deap"]),
            "osa_reduction": 1 - value("osa", baseline["optimized"]) / optimized,
        }

    def _markdown_report(self, checks: pd.DataFrame, headline: Mapping[str, object]) -> str:
        hard_fail = bool(((checks["severity"] == "ERROR") & (checks["status"] == "FAIL")).any())
        warnings = bool((checks["status"] == "WARN").any())
        status = "FAIL" if hard_fail else ("PASS_WITH_PAPER_GAPS" if warnings else "PASS")
        lines = ["# DAC26 EDP Reproduction Report", "", f"Overall status: **{status}**", "",
                 "## Provenance", "", "```json", json.dumps(json.loads((self.run_dir / 'run.json').read_text())["provenance"], indent=2), "```", "",
                 "## Headline metrics", "", "```json", json.dumps(headline, indent=2), "```", "", "## Checks", "",
                 "| Severity | Status | Check | Detail |", "|---|---|---|---|"]
        for row in checks.itertuples(index=False):
            lines.append(f"| {row.severity} | {row.status} | {row.check} | {str(row.detail).replace('|', '/')} |")
        return "\n".join(lines) + "\n"


class ReproductionValidator:
    """Classify simulator invariants as errors and paper gaps as warnings."""

    def __init__(self, manifest: ExperimentManifest, run_dir: Path) -> None:
        self.manifest = manifest
        self.run_dir = Path(run_dir)

    def validate(self, raw: pd.DataFrame, aggregates: pd.DataFrame, headline: Mapping[str, object]) -> pd.DataFrame:
        metadata = json.loads((self.run_dir / "run.json").read_text())
        expected = int(metadata["expected_jobs"])
        rows = [
            self._check("job_coverage", len(raw) == expected, "ERROR", f"{len(raw)}/{expected} successful jobs"),
            self._check("no_duplicate_jobs", raw.empty or raw["job_id"].is_unique, "ERROR", f"{raw['job_id'].nunique() if not raw.empty else 0} unique"),
        ]
        if not raw.empty:
            recomputed = raw.energy_j * raw.latency_s
            relative = (raw.edp_j_s - recomputed).abs() / raw.edp_j_s.abs().clip(lower=1e-30)
            rows.append(self._check("edp_equals_energy_times_latency", bool((relative <= self.manifest.raw["tolerances"]["edp_relative"]).all()), "ERROR", f"max relative error={relative.max():.3e}"))
        if headline.get("available"):
            rows.append(self._reference_check(aggregates))
            rows.append(self._check("feasible_winner", headline["winner"] == self.manifest.raw["baselines"]["optimized"], "ERROR", str(headline["winner"])))
            target_keys = ("optimized_vs_compact_reduction", "optimized_vs_deap_reduction", "osa_reduction")
            tolerance = self.manifest.raw["tolerances"]["paper_percentage_points"]
            for key in target_keys:
                target = float(self.manifest.raw["paper_targets"][key])
                actual = float(headline[key])
                delta = abs(actual - target)
                rows.append({"check": f"paper:{key}", "severity": "WARNING", "status": "PASS" if delta <= tolerance else "WARN", "detail": f"actual={actual:.4%}, paper={target:.4%}, delta={delta:.2%}"})
            rows.append({"check": "paper:optimized_ode", "severity": "WARNING", "status": "WARN", "detail": self.manifest.raw["paper_targets"]["osa_optimized_ode_reduction"]["reason"]})
            rows.append({"check": "paper:hybrid_edp", "severity": "WARNING", "status": "WARN", "detail": self.manifest.raw["paper_targets"]["hybrid_edp_j_s"]["reason"]})
        return pd.DataFrame(rows)

    def _reference_check(self, aggregates: pd.DataFrame) -> dict[str, object]:
        required = {"network", "variant", "cols", "rows", "energy_j", "latency_s", "edp_j_s"}
        if not required.issubset(aggregates.columns):
            return self._check(
                "committed_reference_metrics", False, "ERROR",
                f"aggregate columns missing: {sorted(required - set(aggregates.columns))}",
            )
        reference_dir = self.manifest.repo_root / "examples/rosa/paper_edp_data"
        relative_errors = []
        missing = []
        suffixes = {"no_osa": "", "osa": "_osa"}
        for network in self.manifest.networks:
            for variant, suffix in suffixes.items():
                path = reference_dir / f"aggregated_metrics_{network}_1bit_input{suffix}.csv"
                if not path.exists():
                    missing.append(path.name)
                    continue
                reference = pd.read_csv(path)
                actual = aggregates[(aggregates.network == network) & (aggregates.variant == variant)]
                for row in reference.itertuples(index=False):
                    selected = actual[(actual.cols == row.Cols) & (actual.rows == row.Rows)]
                    if len(selected) != 1:
                        missing.append(f"{network}/{variant}/C{row.Cols}R{row.Rows}")
                        continue
                    for actual_col, reference_value in (
                        ("energy_j", row.Energy), ("latency_s", row.Latency), ("edp_j_s", row.EDP)
                    ):
                        value = float(selected.iloc[0][actual_col])
                        relative_errors.append(abs(value - float(reference_value)) / max(abs(float(reference_value)), 1e-30))
        maximum = max(relative_errors, default=float("inf"))
        tolerance = float(self.manifest.raw["tolerances"]["reference_relative"])
        passed = not missing and maximum <= tolerance
        detail = f"max relative error={maximum:.3%}; missing={len(missing)}"
        return self._check("committed_reference_metrics", passed, "ERROR", detail)

    @staticmethod
    def _check(name: str, passed: bool, severity: str, detail: str) -> dict[str, object]:
        return {"check": name, "severity": severity, "status": "PASS" if passed else "FAIL", "detail": detail}
