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
    AccuracyExperimentConfig,
    LayerSimulator,
    MRRMacroConfig,
    TimeloopBackend,
    TimeloopLayerRef,
    TimeloopMacroConfig,
    TimeloopResultCache,
    LayerManifest,
    LayerPolicy,
    MRRVariationConfig,
)
from opticalloop.accuracy.adapters import ONNSimAccuracyBackend  # noqa: E402
from opticalloop.applications.rosa import (  # noqa: E402
    EnvironmentDoctor,
    ExperimentManifest,
    MultiSliceAnalyzer,
    ReproductionAnalyzer,
    ReproductionRunner,
    run_online_mapping,
)


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
            "--network is required when --workload lacks a network prefix"
        )
    return workload.split("/", 1)[0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


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
                "without --var; choose one form."
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


def _run_accuracy(args) -> None:
    manifest = LayerManifest.load(args.layer_manifest)
    policy = LayerPolicy.load(args.policy, manifest)
    seeds = tuple(args.seed + index for index in range(args.runs))
    experiment = AccuracyExperimentConfig(
        network=manifest.network,
        dataset=manifest.dataset,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        runs=args.runs,
        seeds=seeds,
        variation=MRRVariationConfig(
            thermal_std=args.thermal_std,
            dac_std=args.dac_std,
            thermal_scaling_exponent=args.thermal_scaling_exponent,
            thermal_reference_bits=args.thermal_reference_bits,
        ),
    )
    result = ONNSimAccuracyBackend(args.onnsim_root).run(experiment, policy)
    payload = {
        **result.to_row(),
        "accuracies": result.accuracies,
        "losses": result.losses,
        "seeds": result.seeds,
        "baseline_accuracy": result.baseline_accuracy,
        "metadata": dict(result.metadata),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print("OpticalLoop ONNSim accuracy result")
    print(f"Network:        {result.network}")
    print(f"Scenario:       {result.scenario}")
    print(f"Accuracy:       {result.accuracy_mean:.4f}% +/- {result.accuracy_std:.4f}%")
    print(f"Accuracy delta: {result.accuracy_delta:.4f}%")
    print(f"Wrote:          {args.output}")


def _run_optimize_mapping(args) -> None:
    result = run_online_mapping(args.config, args.output_dir)
    print("OpticalLoop online WS/IS mapping optimization")
    print(f"Status:     {result['status']}")
    print(f"Trials:     {result['trials']}")
    if result["status"] == "success":
        selected = result["selected_result"]
        print(f"Policy:     {result['selected_policy_key']}")
        print(f"Score:      {result['selected_score']:.8f}")
        print(f"EDP:        {selected['edp_j_s']:.6e} J*s")
        print(f"Accuracy:   {selected['accuracy']:.4f}%")
    else:
        print("No candidate satisfied the configured accuracy floor.")
    print(f"Artifacts:  {Path(args.output_dir).resolve()}")


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


def _add_experiment_parser(
    subparsers, *, name: str, help_text: str, manifest: Path,
    run_root: Path, handler,
) -> None:
    """Register the shared native-simulation experiment interface."""
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument(
        "action", choices=("doctor", "smoke", "full", "analyze", "validate")
    )
    parser.add_argument("--manifest", type=Path, default=manifest)
    parser.add_argument("--run-root", type=Path, default=run_root)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-jobs", type=int, default=None,
        help="Run at most this many pending jobs, then stop for deterministic batching.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    parser.set_defaults(func=handler)


def _add_reproduce_parser(subparsers) -> None:
    _add_experiment_parser(
        subparsers,
        name="reproduce",
        help_text="Run and verify the clean-checkout DAC26 EDP experiment.",
        manifest=Path("examples/rosa/dac26_edp_manifest.yaml"),
        run_root=Path("reproduction-runs"),
        handler=_run_reproduce,
    )


def _add_multislice_parser(subparsers) -> None:
    _add_experiment_parser(
        subparsers,
        name="multislice",
        help_text="Run WS/IS multi-slice mapping experiments.",
        manifest=Path("examples/rosa/mb_osa_manifest.yaml"),
        run_root=Path("multislice-runs"),
        handler=_run_multislice,
    )


def _add_accuracy_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "accuracy", help="Run optional whole-network accuracy through ONNSim."
    )
    parser.add_argument("--layer-manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--onnsim-root", type=Path, default=Path("reference/onnsim")
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thermal-std", type=float, default=0.05)
    parser.add_argument("--thermal-scaling-exponent", type=float, default=0.5)
    parser.add_argument(
        "--thermal-reference-bits", type=int, choices=(1, 2, 4, 8), default=8
    )
    parser.add_argument("--dac-std", type=float, default=0.02)
    parser.add_argument(
        "--output", type=Path, default=Path("results/accuracy/onnsim.json")
    )
    parser.set_defaults(func=_run_accuracy)


def _add_optimize_mapping_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "optimize-mapping",
        help="Online WS/IS mapping search using Timeloop EDP and ONNSim accuracy.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/optimization/resnet18_online.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/resnet18"),
    )
    parser.set_defaults(func=_run_optimize_mapping)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Timeloop-backed OpticalLoop simulations and applications."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_layer_parser(subparsers)
    _add_reproduce_parser(subparsers)
    _add_multislice_parser(subparsers)
    _add_accuracy_parser(subparsers)
    _add_optimize_mapping_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
