"""Command-line entrypoint for the OpticalLoop simulator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


def _bootstrap_local_package() -> None:
    package_root = Path(__file__).resolve().parent
    if "opticalloop" in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        "opticalloop",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load opticalloop package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["opticalloop"] = module
    spec.loader.exec_module(module)


_bootstrap_local_package()

from opticalloop import (  # noqa: E402
    LayerSimulator,
    MRRMacroConfig,
    TimeloopBackend,
    TimeloopLayerRef,
    TimeloopMacroConfig,
    TimeloopResultCache,
)
from opticalloop.applications.rosa import (  # noqa: E402
    FINAL_ARTIFACTS,
    RosaResultValidator,
    PaperEDPConfig,
    PaperEDPReproduction,
    EnvironmentDoctor,
    ExperimentManifest,
    ReproductionAnalyzer,
    ReproductionRunner,
    MultiSliceAnalyzer,
    default_rosa_workflow,
    parse_architecture_argument,
    write_reference_artifacts,
)


def _parse_networks(value: str) -> Sequence[str]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_cli_value(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_variable_assignments(assignments: Sequence[str]) -> dict[str, object]:
    variables: dict[str, object] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise SystemExit(f"--var must use KEY=VALUE syntax: {assignment!r}")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--var key cannot be empty: {assignment!r}")
        variables[key] = _parse_cli_value(value.strip())
    return variables


def _infer_network_from_workload(workload: str) -> str:
    if "/" not in workload:
        raise SystemExit(
            "--network is required when --workload does not include a network prefix"
        )
    return workload.split("/", 1)[0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_path_within_repo(path: Path, *, label: str) -> Path:
    repo_root = _repo_root().resolve()
    resolved = Path(path).resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise SystemExit(
            f"{label} must stay inside the OpticalLoop repository: {path.as_posix()}"
        )
    return path


def _source_has_artifacts(path: Path) -> bool:
    return all(
        (path / source_relative_path).exists() or (path / output_name).exists()
        for output_name, source_relative_path in FINAL_ARTIFACTS.items()
    )


def _default_artifact_source_results_dir() -> Path:
    for candidate in (
        Path("results"),
        Path("examples/rosa/results"),
    ):
        if _source_has_artifacts(candidate):
            return candidate
    return Path("results")


def _has_full_cache_report(results_dir: Path) -> bool:
    return (
        results_dir
        / "reconstructed"
        / "aggregated_metrics_alexnet_1bit_input_osa.csv"
    ).exists()


def _artifact_cache_report(results_dir: Path):
    alexnet_path = results_dir / "aggregated_metrics_alexnet_1bit_input_osa.csv"
    ranking_path = (
        results_dir
        / "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv"
    )
    if not alexnet_path.exists() or not ranking_path.exists():
        missing = alexnet_path if not alexnet_path.exists() else ranking_path
        raise FileNotFoundError(missing)

    alexnet = pd.read_csv(alexnet_path)
    best_osa = alexnet.sort_values("EDP", kind="mergesort").iloc[0]
    ranking = pd.read_csv(ranking_path)
    best_ranking = ranking.sort_values("Rank", kind="mergesort").iloc[0]
    return {
        "alexnet_osa_best": {
            "architecture": (
                f"T{int(best_osa['Tiles'])},P{int(best_osa['PEs'])},"
                f"C{int(best_osa['Cols'])},R{int(best_osa['Rows'])}"
            ),
            "edp": float(best_osa["EDP"]),
            "energy": float(best_osa["Energy"]),
            "latency": float(best_osa["Latency"]),
        },
        "six_network_osa_best": {
            "architecture": str(best_ranking["Architecture"]),
            "aggregated_score": float(best_ranking["Aggregated_Score"]),
            "rank": int(best_ranking["Rank"]),
        },
    }


def _print_rosa_report(report) -> None:
    alexnet = report["alexnet_osa_best"]
    ranking = report["six_network_osa_best"]
    print("OpticalLoop ROSA cached report")
    print(
        "AlexNet OSA best: "
        f"{alexnet['architecture']} EDP={alexnet['edp']:.10f} "
        f"Energy={alexnet['energy']:.6e} Latency={alexnet['latency']:.6e}"
    )
    print(
        "Six-network OSA best: "
        f"{ranking['architecture']} score={ranking['aggregated_score']:.10f} "
        f"rank={ranking['rank']}"
    )


def _write_validation_report(results_dir: Path, report_path: Path) -> None:
    validator = RosaResultValidator(results_dir)
    report = validator.validate()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    failed = report[~report["passed"]]
    if not failed.empty:
        raise SystemExit(
            "Validation failed; see "
            f"{report_path.as_posix()} for {len(failed)} failing checks"
        )
    print(f"Validation passed ({len(report)} checks); wrote {report_path.as_posix()}")


def _run_rosa(args) -> None:
    workflow = default_rosa_workflow(
        results_dir=args.results_dir,
        networks=_parse_networks(args.networks),
        n_jobs=args.n_jobs,
    )

    if args.mode == "cache" and args.stage in {"run", "hybrid"}:
        raise SystemExit("cache mode cannot run live Timeloop stages; use --mode rerun")

    if args.stage in {"paper-edp", "all"}:
        reproduction = PaperEDPReproduction(
            PaperEDPConfig(data_dir=args.paper_edp_data_dir)
        )
        summary = reproduction.headline_summary()
        print("DAC26 EDP-only reproduction (committed Timeloop aggregates)")
        print(f"No-OSA winner: {summary['best_no_osa']}")
        print(
            "Optimized reduction vs compact / DEAP: "
            f"{summary['optimized_vs_compact_reduction']:.1%} / "
            f"{summary['optimized_vs_deap_reduction']:.1%}"
        )
        print(
            "OSA reduction at optimized shape: "
            f"{summary['osa_reduction_at_optimized_shape']:.1%}"
        )
        print("Paper targets: 26% / 64% / 29% (37% with optimized ODE)")

    if args.stage in {"run", "all"} and args.mode == "rerun":
        workflow.run_sweeps()

    if args.stage in {"aggregate", "all"}:
        workflow.reconstruct_and_aggregate()

    if args.stage in {"rank", "all"}:
        ranking = workflow.rank_osa_architectures()
        best = ranking.iloc[0]
        print(
            "Wrote OSA ranking; best architecture: "
            f"{best['Architecture']} score={best['Aggregated_Score']:.10f}"
        )

    if args.stage in {"hybrid", "all"} and args.mode == "rerun":
        architecture = parse_architecture_argument(args.hybrid_architecture)
        outputs = workflow.run_hybrid(
            family=args.hybrid_family,
            architecture=architecture,
        )
        print(f"Wrote {len(outputs)} hybrid workflow artifact groups")

    if args.stage in {"report", "all"}:
        if _has_full_cache_report(workflow.results_dir):
            _print_rosa_report(workflow.cache_report())
        else:
            _print_rosa_report(_artifact_cache_report(args.artifact_results_dir))

    if args.stage in {"artifacts", "all"}:
        source_results_dir = (
            args.artifact_source_results_dir or _default_artifact_source_results_dir()
        )
        source_results_dir = _ensure_path_within_repo(
            source_results_dir,
            label="--artifact-source-results-dir",
        )
        outputs = write_reference_artifacts(
            source_results_dir=source_results_dir,
            output_results_dir=args.artifact_results_dir,
            output_plots_dir=args.artifact_plots_dir,
        )
        print(
            "Wrote lightweight artifacts from "
            f"{source_results_dir.as_posix()}: {len(outputs)} files"
        )

    if args.stage in {"validate", "all"}:
        report_path = args.validation_report or (
            args.artifact_results_dir / "validation_report.csv"
        )
        _write_validation_report(args.artifact_results_dir, report_path)


def _run_layer(args) -> None:
    variables = _parse_variable_assignments(args.var or ())
    workload = args.workload or args.layer
    if workload is None:
        raise SystemExit("Provide --workload, or legacy --layer with --network")
    network = args.network or _infer_network_from_workload(workload)

    uses_mrr_shape = all(
        value is not None for value in (args.tiles, args.pes, args.cols, args.rows)
    )
    if uses_mrr_shape and not variables:
        architecture = MRRMacroConfig(
            n_tiles=args.tiles,
            n_pes=args.pes,
            n_cols=args.cols,
            n_rows=args.rows,
            macro=args.arch,
            system=args.system,
            front_mrr_slice_bits=args.front_mrr_slice_bits,
            scaling=args.scaling,
            max_utilization=args.max_utilization,
        )
    else:
        if any(value is not None for value in (args.tiles, args.pes, args.cols, args.rows)):
            raise SystemExit(
                "Use either --var for a generic macro or all MRR shape options "
                "without --var; do not mix the two forms."
            )
        architecture = TimeloopMacroConfig(
            macro=args.arch,
            variables=variables,
            system=args.system,
            max_utilization=args.max_utilization,
        )

    layer = TimeloopLayerRef(network=network, layer_path=workload)
    cache = None
    if args.cache_results_dir is not None:
        if not isinstance(architecture, MRRMacroConfig):
            raise SystemExit("--cache-results-dir is only supported for MRR shape runs")
        cache = TimeloopResultCache(
            results_dir=args.cache_results_dir,
            save_name=args.cache_save_name,
            output_postfix=args.cache_output_postfix,
        )
    result = LayerSimulator(
        layer=layer,
        architecture=architecture,
        backend=TimeloopBackend(),
        cache=cache,
    ).run()

    print("OpticalLoop Timeloop-backed layer result")
    print(f"Source:       {result.source}")
    print(f"Workload:     {layer.layer_path}")
    print(f"Architecture: {architecture.architecture_key}")
    print(f"Macro:        {architecture.macro}")
    print(f"Cycles:       {result.cycles}")
    print(f"Latency:      {result.latency_s:.6e} s")
    print(f"Energy:       {result.energy_j:.6e} J")
    if result.area_mm2 is not None:
        print(f"Area:         {result.area_mm2:.6e} mm^2")
    if result.tops_per_w is not None:
        print(f"TOPS/W:       {result.tops_per_w:.6f}")
    if args.show_mapping:
        print("Mapping:")
        print(result.mapping_text or "<no Timeloop mapping text available>")


def _report_incomplete_batch(run_dir: Path) -> bool:
    metadata = json.loads((Path(run_dir) / "run.json").read_text())
    if metadata.get("status") != "incomplete":
        return False
    print(f"Run directory: {Path(run_dir).resolve()}")
    print(
        "Batch complete: "
        f"{metadata.get('successful_jobs', 0)}/{metadata.get('expected_jobs', 0)} "
        f"successful; {metadata.get('remaining_jobs', 0)} pending. "
        "Repeat the same command to resume."
    )
    return True


def _run_reproduce(args) -> None:
    manifest = ExperimentManifest(args.manifest, repo_root=_repo_root())
    doctor = EnvironmentDoctor(manifest)
    if args.action == "doctor":
        report = doctor.check()
        print(report.to_string(index=False))
        if (report["status"] == "FAIL").any():
            raise SystemExit(2)
        return

    if args.action in {"smoke", "full"}:
        preflight = doctor.check()
        failed = preflight[preflight["status"] == "FAIL"]
        if not failed.empty:
            print(failed.to_string(index=False))
            raise SystemExit("Environment doctor failed; simulation was not started")
        run_dir = ReproductionRunner(manifest, args.run_root).run(
            args.action, resume=not args.no_resume, fail_fast=args.fail_fast,
            workers=args.workers, max_jobs=args.max_jobs,
        )
        if _report_incomplete_batch(run_dir):
            return
    else:
        if args.run_dir is None:
            raise SystemExit(f"reproduce {args.action} requires --run-dir")
        run_dir = args.run_dir

    artifacts = ReproductionAnalyzer(manifest, run_dir).analyze(
        execute_notebook=args.action != "validate" and not args.skip_notebook
    )
    print(f"Run directory: {Path(run_dir).resolve()}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    validation = pd.read_csv(artifacts["validation.csv"])
    if ((validation["severity"] == "ERROR") & (validation["status"] == "FAIL")).any():
        raise SystemExit(1)


def _run_multislice(args) -> None:
    manifest = ExperimentManifest(args.manifest, repo_root=_repo_root())
    doctor = EnvironmentDoctor(manifest)
    if args.action == "doctor":
        report = doctor.check()
        print(report.to_string(index=False))
        if (report["status"] == "FAIL").any():
            raise SystemExit(2)
        return

    if args.action in {"smoke", "full"}:
        preflight = doctor.check()
        failed = preflight[preflight.status == "FAIL"]
        if not failed.empty:
            print(failed.to_string(index=False))
            raise SystemExit("Environment doctor failed; simulation was not started")
        run_dir = ReproductionRunner(manifest, args.run_root).run(
            args.action,
            resume=not args.no_resume,
            fail_fast=args.fail_fast,
            workers=args.workers, max_jobs=args.max_jobs,
        )
        if _report_incomplete_batch(run_dir):
            return
    else:
        if args.run_dir is None:
            raise SystemExit(f"multislice {args.action} requires --run-dir")
        run_dir = args.run_dir

    outputs = MultiSliceAnalyzer(manifest, run_dir).analyze(
        execute_notebook=args.action != "validate" and not args.skip_notebook
    )
    print(f"Run directory: {Path(run_dir).resolve()}")
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")
    validation = pd.read_csv(outputs["validation.csv"])
    if ((validation.severity == "ERROR") & (validation.status == "FAIL")).any():
        raise SystemExit(1)


def _add_rosa_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "rosa",
        help="Run the ROSA application workflow.",
    )
    parser.add_argument("--mode", choices=("cache", "rerun"), default="cache")
    parser.add_argument(
        "--stage",
        choices=(
            "report",
            "run",
            "aggregate",
            "rank",
            "hybrid",
            "validate",
            "artifacts",
            "paper-edp",
            "all",
        ),
        default="report",
    )
    parser.add_argument("--preset", choices=("rosa-full",), default="rosa-full")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--paper-edp-data-dir",
        type=Path,
        default=Path("examples/rosa/paper_edp_data"),
        help="Committed six-workload no-OSA/OSA Timeloop aggregate directory.",
    )
    parser.add_argument(
        "--networks",
        type=str,
        default="alexnet,vgg16,resnet18,mobilenet_v3,gpt2_medium,vision_transformer",
    )
    parser.add_argument("--n-jobs", type=int, default=128)
    parser.add_argument(
        "--hybrid-family",
        choices=("osa", "delay-line", "both"),
        default="both",
    )
    parser.add_argument(
        "--hybrid-architecture",
        type=str,
        default="T1, P16, C8, R8",
        help="Hybrid architecture as 'T1, P16, C8, R8' or '1,16,8,8'.",
    )
    parser.add_argument(
        "--artifact-source-results-dir",
        type=Path,
        default=None,
        help=(
            "In-repo source Timeloop result tree for lightweight artifacts. "
            "Defaults to results, then examples/rosa/results."
        ),
    )
    parser.add_argument(
        "--artifact-results-dir",
        type=Path,
        default=Path("examples/rosa/results"),
        help="Portable ROSA CSV artifact output directory.",
    )
    parser.add_argument(
        "--artifact-plots-dir",
        type=Path,
        default=Path("examples/rosa/plots"),
        help="Portable ROSA PNG artifact output directory.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help="Validation report path. Defaults to <artifact-results-dir>/validation_report.csv.",
    )
    parser.set_defaults(func=_run_rosa)


def _add_layer_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "layer",
        help="Run one Timeloop-backed layer simulation.",
    )
    parser.add_argument(
        "--arch",
        "--macro",
        dest="arch",
        default="mrr_ws_osa",
        help="Timeloop macro/architecture name.",
    )
    parser.add_argument(
        "--workload",
        default=None,
        help="Timeloop workload path such as alexnet/0 or deap_deepbench/bench0.",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Optional network label. Inferred from --workload when possible.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Legacy alias for --workload when using MRR shape options.",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="Timeloop variable override in KEY=VALUE form. Repeat as needed.",
    )
    parser.add_argument("--system", default="fetch_all_lpddr4")
    parser.add_argument("--tiles", type=int, default=None)
    parser.add_argument("--pes", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument(
        "--front-mrr-slice-bits",
        dest="front_mrr_slice_bits", type=int, choices=(1, 2, 4, 8), default=1,
        help="Bits per temporal symbol at the first MRR modulation stage.",
    )
    parser.add_argument("--scaling", default='"aggressive"')
    parser.add_argument("--max-utilization", action="store_true")
    parser.add_argument(
        "--show-mapping",
        action="store_true",
        help="Print Timeloop mapper loop text when available.",
    )
    parser.add_argument("--cache-results-dir", type=Path, default=None)
    parser.add_argument("--cache-save-name", default="deapcnns")
    parser.add_argument("--cache-output-postfix", default="_1bit_input_osa")
    parser.set_defaults(func=_run_layer)


def _add_reproduce_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "reproduce", help="Run and verify the clean-checkout DAC26 EDP experiment."
    )
    parser.add_argument("action", choices=("doctor", "smoke", "full", "analyze", "validate"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("examples/rosa/dac26_edp_manifest.yaml"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("reproduction-runs"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-jobs", type=int, default=None,
        help="Run at most this many pending jobs, then stop cleanly for deterministic batching.",
    )
    parser.add_argument("--skip-notebook", action="store_true")
    parser.set_defaults(func=_run_reproduce)


def _add_multislice_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "multislice", help="Run MB-OSA and Adaptive Slice-Width Mapping experiments."
    )
    parser.add_argument("action", choices=("doctor", "smoke", "full", "analyze", "validate"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("examples/rosa/mb_osa_manifest.yaml")
    )
    parser.add_argument("--run-root", type=Path, default=Path("multislice-runs"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-jobs", type=int, default=None,
        help="Run at most this many pending jobs, then stop cleanly for deterministic batching.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    parser.set_defaults(func=_run_multislice)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Timeloop-backed OpticalLoop simulations and applications."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_layer_parser(subparsers)
    _add_rosa_parser(subparsers)
    _add_reproduce_parser(subparsers)
    _add_multislice_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
