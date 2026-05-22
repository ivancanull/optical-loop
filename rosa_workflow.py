"""Command-line entrypoint for the OpticalLoop ROSA workflow."""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or report the Timeloop-backed OpticalLoop ROSA workflow."
    )
    parser.add_argument("--mode", choices=("cache", "rerun"), default="cache")
    parser.add_argument(
        "--stage",
        choices=("report", "run", "aggregate", "rank", "hybrid", "all"),
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
        _print_report(workflow.cache_report())


if __name__ == "__main__":
    main()
