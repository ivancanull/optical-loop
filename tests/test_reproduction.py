import json
from pathlib import Path

import pandas as pd
import pytest

from opticalloop.applications.rosa.reproduction import (
    EnvironmentDoctor,
    ExperimentManifest,
    ReproductionAnalyzer,
    ReproductionRunner,
    ReproductionValidator,
)
from opticalloop.result import SimulationResult


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def run_layer(self, layer, architecture) -> SimulationResult:
        self.calls += 1
        scale = architecture.n_cols * architecture.n_rows
        return SimulationResult(
            cycles=scale,
            latency_s=scale * 1e-9,
            energy_j=scale * 1e-12,
            energy_breakdown={"fake": scale * 1e-12},
            area_mm2=1e-3,
            compute=float(scale),
            cycle_seconds=1e-9,
            tops=1.0,
        )


@pytest.fixture
def manifest() -> ExperimentManifest:
    root = Path(__file__).resolve().parents[1]
    return ExperimentManifest(
        root / "examples/rosa/dac26_edp_manifest.yaml", repo_root=root
    )


def test_manifest_expands_expected_smoke_and_full_jobs(manifest) -> None:
    assert sum(len(manifest.layers(network)) for network in manifest.networks) == 352
    assert len(manifest.jobs("smoke")) == 12
    assert len(manifest.jobs("full")) == 7040
    assert len({job.job_id for job in manifest.jobs("full")}) == 7040


def test_manifest_enforces_candidate_constraints(manifest, tmp_path: Path) -> None:
    raw = dict(manifest.raw)
    raw["architectures"] = [dict(value) for value in manifest.raw["architectures"]]
    raw["architectures"][2]["cols"] = 9
    path = tmp_path / "manifest.yaml"
    path.write_text(__import__("yaml").safe_dump(raw))
    with pytest.raises(ValueError, match="column constraint"):
        ExperimentManifest(path, repo_root=manifest.repo_root)


def test_smoke_run_checkpoints_and_resumes(manifest, tmp_path: Path) -> None:
    backend = FakeBackend()
    runner = ReproductionRunner(manifest, tmp_path, backend=backend)
    run_dir = runner.run("smoke", workers=2)
    assert backend.calls == 12
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["successful_jobs"] == 12
    assert len(list((run_dir / "logs").glob("*.log"))) == 12

    runner.run("smoke", workers=2)
    assert backend.calls == 12


def test_smoke_analysis_labels_headlines_unavailable(manifest, tmp_path: Path) -> None:
    run_dir = ReproductionRunner(manifest, tmp_path, backend=FakeBackend()).run(
        "smoke", workers=2
    )
    artifacts = ReproductionAnalyzer(manifest, run_dir).analyze()
    headline = json.loads(artifacts["headline.json"].read_text())
    checks = pd.read_csv(artifacts["validation.csv"])
    assert not headline["available"]
    assert checks["status"].eq("PASS").all()
    assert "Overall status: **PASS**" in artifacts["REPORT.md"].read_text()


def test_analysis_detects_missing_and_corrupt_jobs(manifest, tmp_path: Path) -> None:
    run_dir = ReproductionRunner(manifest, tmp_path, backend=FakeBackend()).run(
        "smoke", workers=1
    )
    first = next((run_dir / "jobs").glob("*.json"))
    payload = json.loads(first.read_text())
    payload["metrics"]["edp_j_s"] *= 2
    first.write_text(json.dumps(payload))
    checks = pd.read_csv(ReproductionAnalyzer(manifest, run_dir).analyze()["validation.csv"])
    failed = set(checks.loc[checks.status == "FAIL", "check"])
    assert "edp_equals_energy_times_latency" in failed

    first.unlink()
    checks = pd.read_csv(ReproductionAnalyzer(manifest, run_dir).analyze()["validation.csv"])
    assert "job_coverage" in set(checks.loc[checks.status == "FAIL", "check"])


def test_doctor_reports_absent_native_tools(manifest, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    report = EnvironmentDoctor(manifest).check().set_index("check")
    assert report.loc["executable:timeloop-mapper", "status"] == "FAIL"
    assert report.loc["accelergy:plugin_discovery", "status"] == "FAIL"


def test_resume_rejects_changed_provenance(manifest, tmp_path: Path) -> None:
    runner = ReproductionRunner(manifest, tmp_path, backend=FakeBackend())
    run_dir = runner.run("smoke", workers=1)
    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance"]["python"] = "different"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="different toolchain"):
        runner.run("smoke", workers=1)


def test_validator_rejects_unexpected_full_sweep_winner(manifest, tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"expected_jobs": 0, "tier": "full"}))
    headline = {
        "available": True,
        "winner": "compact_4x4",
        "optimized_vs_compact_reduction": 0.26,
        "optimized_vs_deap_reduction": 0.64,
        "osa_reduction": 0.29,
    }
    checks = ReproductionValidator(manifest, tmp_path).validate(
        pd.DataFrame(), pd.DataFrame(), headline
    )
    winner = checks[checks.check == "feasible_winner"].iloc[0]
    assert winner.status == "FAIL"
    assert winner.severity == "ERROR"
