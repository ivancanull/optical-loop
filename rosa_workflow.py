"""Command-line entrypoint for the OpticalLoop ROSA workflow."""

import argparse
import importlib.util
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

from opticalloop.workflow import default_rosa_workflow
from opticalloop.workflow.rosa import parse_architecture_argument
from opticalloop.workflow.validation import (
    FINAL_ARTIFACTS,
    RosaResultValidator,
    write_reference_artifacts,
)


def _parse_networks(value: str) -> Sequence[str]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _print_report(report) -> None:
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


def _source_has_artifacts(path: Path) -> bool:
    return all(
        (path / source_relative_path).exists() or (path / output_name).exists()
        for output_name, source_relative_path in FINAL_ARTIFACTS.items()
    )


def _default_artifact_source_results_dir() -> Path:
    for candidate in (Path("results"), Path("../results"), Path("examples/results")):
        if _source_has_artifacts(candidate):
            return candidate
    return Path("results")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or report the Timeloop-backed OpticalLoop ROSA workflow."
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
            "all",
        ),
        default="report",
    )
    parser.add_argument("--preset", choices=("rosa-full",), default="rosa-full")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
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
            "Source Timeloop/CIMLoop result tree for lightweight artifacts. "
            "Defaults to results, ../results, then examples/results when available."
        ),
    )
    parser.add_argument(
        "--artifact-results-dir",
        type=Path,
        default=Path("examples/results"),
        help="Portable CSV artifact output directory.",
    )
    parser.add_argument(
        "--artifact-plots-dir",
        type=Path,
        default=Path("examples/plots"),
        help="Portable PNG artifact output directory.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help="Validation report path. Defaults to <artifact-results-dir>/validation_report.csv.",
    )
    args = parser.parse_args()

    _ = args.preset
    workflow = default_rosa_workflow(
        results_dir=args.results_dir,
        networks=_parse_networks(args.networks),
        n_jobs=args.n_jobs,
    )

    if args.mode == "cache" and args.stage in {"run", "hybrid"}:
        raise SystemExit("cache mode cannot run live Timeloop stages; use --mode rerun")

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
            _print_report(workflow.cache_report())
        else:
            _print_report(_artifact_cache_report(args.artifact_results_dir))

    if args.stage in {"artifacts", "all"}:
        source_results_dir = (
            args.artifact_source_results_dir or _default_artifact_source_results_dir()
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


if __name__ == "__main__":
    main()
