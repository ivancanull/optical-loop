"""Timeloop-backed ROSA application workflow.

This module intentionally performs orchestration and CSV post-processing only.
All live mapper execution goes through `TimeloopBackend`.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from opticalloop.backend import TimeloopBackend, TimeloopRun
from opticalloop.cache import parse_architecture_key
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.module_data import write_module_data_csv as write_module_rows_csv
from opticalloop.result import SimulationResult
from opticalloop.workflow.results import (
    ArtifactPaths,
    aggregate_metrics_csv,
    artifact_paths,
    reconstruct_breakdown_csv,
    write_architecture_metrics_csv as write_architecture_metrics_rows_csv,
    write_results_csv,
)


DEFAULT_ROSA_NETWORKS: Tuple[str, ...] = (
    "alexnet",
    "vgg16",
    "resnet18",
    "mobilenet_v3",
    "gpt2_medium",
    "vision_transformer",
)

DEFAULT_ARCHITECTURES: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 1, 9, 113),
    (1, 1, 100, 12),
    (1, 64, 4, 4),
    (1, 32, 4, 8),
    (1, 16, 4, 16),
    (1, 8, 4, 32),
    (1, 32, 8, 4),
    (1, 16, 8, 8),
    (1, 8, 8, 16),
    (1, 4, 8, 32),
)

COMPONENT_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "MRM": ("weight_mrr", "input_mrr", "mrr"),
    "DAC": ("weight_dac", "input_dac"),
    "OAC": ("laser", "photodiode_output_readout", "TIA"),
    "ADC": ("adc",),
    "Cache": ("glb", "output_buffer"),
    "Main Memory": ("main_memory",),
}

ARCH_RE = re.compile(
    r"T(?P<Tiles>\d+),\s*P(?P<PEs>\d+),\s*C(?P<Cols>\d+),\s*R(?P<Rows>\d+)"
)


@dataclass(frozen=True)
class ArchitectureSetting:
    """MRR macro shape used in the ROSA sweeps."""

    n_tiles: int
    n_pes: int
    n_cols: int
    n_rows: int

    def __post_init__(self) -> None:
        for name, value in (
            ("n_tiles", self.n_tiles),
            ("n_pes", self.n_pes),
            ("n_cols", self.n_cols),
            ("n_rows", self.n_rows),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def filename_key(self) -> str:
        return (
            f"{self.n_tiles}tiles_"
            f"{self.n_pes}pes_"
            f"{self.n_cols}cols_"
            f"{self.n_rows}rows"
        )

    @property
    def architecture_key(self) -> str:
        return f"T{self.n_tiles}, P{self.n_pes}, C{self.n_cols}, R{self.n_rows}"

    def to_macro_config(
        self,
        variant: "MacroVariant",
        *,
        system: str,
    ) -> MRRMacroConfig:
        return MRRMacroConfig(
            n_tiles=self.n_tiles,
            n_pes=self.n_pes,
            n_cols=self.n_cols,
            n_rows=self.n_rows,
            macro=variant.macro,
            system=system,
            voltage_dac_resolution=variant.voltage_dac_resolution,
            scaling=variant.scaling,
            max_utilization=variant.max_utilization,
        )


@dataclass(frozen=True)
class MacroVariant:
    """One macro flavor in a ROSA reproduction sweep."""

    macro: str
    label: str
    output_postfix: str
    voltage_dac_resolution: int = 1
    max_utilization: bool = False
    scaling: str = '"aggressive"'

    def __post_init__(self) -> None:
        if not self.macro:
            raise ValueError("macro cannot be empty")
        if not self.label:
            raise ValueError("label cannot be empty")
        if self.output_postfix is None:
            raise ValueError("output_postfix cannot be None")
        if self.voltage_dac_resolution <= 0:
            raise ValueError("voltage_dac_resolution must be positive")
        if not self.scaling:
            raise ValueError("scaling cannot be empty")


@dataclass(frozen=True)
class HybridMappingSpec:
    """Layer-wise macro selection for ROSA hybrid mapping."""

    network: str
    mapping_path: Path
    regular_macro: str
    hybrid_macro: str
    architecture: ArchitectureSetting = ArchitectureSetting(1, 16, 8, 8)
    output_postfix: str = "hybrid_mapping"
    save_name: str = "hybrid_mapping"
    voltage_dac_resolution: int = 1
    max_utilization: bool = False
    scaling: str = '"aggressive"'
    family: str = "osa"

    def __post_init__(self) -> None:
        if not self.network:
            raise ValueError("network cannot be empty")
        if not self.regular_macro or not self.hybrid_macro:
            raise ValueError("hybrid macros cannot be empty")

    def load_mapping(self) -> Dict[str, bool]:
        if not self.mapping_path.exists():
            raise FileNotFoundError(self.mapping_path)
        with self.mapping_path.open() as mapping_file:
            raw = yaml.safe_load(mapping_file) or {}
        return {str(key): bool(value) for key, value in raw.items()}

    def macro_for_layer(self, layer_id: str) -> str:
        mapping = self.load_mapping()
        return self.hybrid_macro if mapping.get(str(layer_id), False) else self.regular_macro

    def variant_for_layer(self, layer_id: str) -> MacroVariant:
        macro = self.macro_for_layer(layer_id)
        return MacroVariant(
            macro=macro,
            label=f"{self.family}:{macro}",
            output_postfix=self.output_postfix,
            voltage_dac_resolution=self.voltage_dac_resolution,
            max_utilization=self.max_utilization,
            scaling=self.scaling,
        )


@dataclass(frozen=True)
class RosaWorkflowSpec:
    """Configuration for a ROSA reproduction workflow."""

    networks: Tuple[str, ...]
    macro_variants: Tuple[MacroVariant, ...]
    architecture_settings: Tuple[ArchitectureSetting, ...]
    system: str = "fetch_all_lpddr4"
    save_name: str = "deapcnns"
    results_dir: Path = Path("results")
    n_jobs: int = 128
    repo_root: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.networks:
            raise ValueError("networks cannot be empty")
        if not self.macro_variants:
            raise ValueError("macro_variants cannot be empty")
        if not self.architecture_settings:
            raise ValueError("architecture_settings cannot be empty")
        if not self.system:
            raise ValueError("system cannot be empty")
        if not self.save_name:
            raise ValueError("save_name cannot be empty")
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive")


class RosaWorkflow:
    """Run and analyze Timeloop-backed ROSA reproduction artifacts."""

    def __init__(
        self,
        spec: RosaWorkflowSpec,
        backend: Optional[TimeloopBackend] = None,
    ) -> None:
        self.spec = spec
        self.backend = backend or TimeloopBackend()
        self.repo_root = (spec.repo_root or _default_repo_root()).resolve()
        self.results_dir = _resolve_path(self.repo_root, spec.results_dir)
        self.workspace_dir = _default_workspace_dir(self.repo_root)
        self.models_dir = self.workspace_dir / "models"

    def run_sweeps(self) -> Dict[Tuple[str, str], ArtifactPaths]:
        outputs: Dict[Tuple[str, str], ArtifactPaths] = {}
        for network in self.spec.networks:
            layers = self.resolve_layers(network)
            for variant in self.spec.macro_variants:
                results = self._run_network_variant(network, layers, variant)
                paths = artifact_paths(
                    save_name=self.spec.save_name,
                    network=network,
                    output_postfix=variant.output_postfix,
                    results_dir=self.results_dir,
                )
                self.write_breakdown_csv(results, paths.breakdown_csv)
                self.write_combined_csv(results, paths.combined_csv)
                self.write_module_data_csv(results, paths.module_data_csv)
                self.write_architecture_metrics_csv(results, paths.metrics_csv)
                outputs[(network, variant.label)] = paths
        return outputs

    def reconstruct_and_aggregate(self) -> None:
        for network in self.spec.networks:
            for variant in self.spec.macro_variants:
                paths = artifact_paths(
                    save_name=self.spec.save_name,
                    network=network,
                    output_postfix=variant.output_postfix,
                    results_dir=self.results_dir,
                )
                reconstruct_breakdown_csv(paths.breakdown_csv, paths.reconstructed_breakdown_csv)
                aggregate_metrics_csv(
                    paths.reconstructed_breakdown_csv, paths.aggregated_metrics_csv
                )

    def rank_osa_architectures(
        self,
        networks: Sequence[str] = DEFAULT_ROSA_NETWORKS,
        output_postfix: str = "_1bit_input_osa",
        alpha: float = 1.0,
        beta: float = 1.5,
        lambda_worst_case: float = 0.25,
    ) -> pd.DataFrame:
        _, detailed, ranking = score_networks(
            networks=tuple(networks),
            metrics_dir=self.results_dir,
            save_name=self.spec.save_name,
            output_postfix=output_postfix,
            alpha=alpha,
            beta=beta,
            lambda_worst_case=lambda_worst_case,
        )
        ranking_path = (
            self.results_dir
            / "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv"
        )
        detailed_path = (
            self.results_dir
            / "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv"
        )
        ranking.to_csv(ranking_path, index=False)
        detailed.to_csv(detailed_path, index=False)
        return ranking

    def cache_report(self) -> Dict[str, Mapping[str, object]]:
        alexnet_osa = self.results_dir / "reconstructed" / (
            "aggregated_metrics_alexnet_1bit_input_osa.csv"
        )
        if not alexnet_osa.exists():
            raise FileNotFoundError(alexnet_osa)
        alexnet_df = pd.read_csv(alexnet_osa)
        best_osa = alexnet_df.sort_values("EDP", kind="mergesort").iloc[0]

        ranking_path = (
            self.results_dir
            / "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv"
        )
        if ranking_path.exists():
            ranking = pd.read_csv(ranking_path)
        else:
            ranking = self.rank_osa_architectures()
        best_ranking = ranking.sort_values("Rank", kind="mergesort").iloc[0]

        report = {
            "alexnet_osa_best": {
                "architecture": _row_architecture(best_osa),
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
        return report

    def run_hybrid(
        self,
        family: str = "both",
        networks: Optional[Sequence[str]] = None,
        architecture: ArchitectureSetting = ArchitectureSetting(1, 16, 8, 8),
    ) -> Dict[Tuple[str, str], ArtifactPaths]:
        outputs: Dict[Tuple[str, str], ArtifactPaths] = {}
        for hybrid_spec in self.default_hybrid_specs(
            family=family,
            networks=networks,
            architecture=architecture,
        ):
            layers = self.resolve_layers(hybrid_spec.network)
            results = self._run_hybrid_spec(hybrid_spec, layers)
            paths = artifact_paths(
                save_name=hybrid_spec.save_name,
                network=hybrid_spec.network,
                output_postfix=hybrid_spec.output_postfix,
                results_dir=self.results_dir,
            )
            self.write_breakdown_csv(results, paths.breakdown_csv)
            self.write_combined_csv(results, paths.combined_csv)
            self.write_module_data_csv(results, paths.module_data_csv)
            self.write_architecture_metrics_csv(results, paths.metrics_csv)
            self._write_hybrid_aliases(hybrid_spec, paths)
            outputs[(hybrid_spec.network, hybrid_spec.family)] = paths
        return outputs

    def default_hybrid_specs(
        self,
        family: str = "both",
        networks: Optional[Sequence[str]] = None,
        architecture: ArchitectureSetting = ArchitectureSetting(1, 16, 8, 8),
    ) -> Tuple[HybridMappingSpec, ...]:
        selected_networks = tuple(networks or self.spec.networks)
        family_names = ("osa", "delay-line") if family == "both" else (family,)
        specs: List[HybridMappingSpec] = []
        for network in selected_networks:
            mapping_path = self.workspace_dir / "hybrid_mapping" / f"{network}.yaml"
            if not mapping_path.exists():
                continue
            for family_name in family_names:
                regular, hybrid = _hybrid_family_macros(family_name)
                specs.append(
                    HybridMappingSpec(
                        network=network,
                        mapping_path=mapping_path,
                        regular_macro=regular,
                        hybrid_macro=hybrid,
                        architecture=architecture,
                        family=family_name,
                    )
                )
        return tuple(specs)

    def resolve_layers(self, network: str) -> Tuple[str, ...]:
        workload_dir = self.models_dir / "workloads" / network
        if not workload_dir.exists():
            raise FileNotFoundError(workload_dir)
        return tuple(
            f"{network}/{path.stem}" for path in sorted(workload_dir.glob("*.yaml"))
        )

    def write_breakdown_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_results_csv(results, path, group_components=False)

    def write_combined_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_results_csv(
            results,
            path,
            group_components=True,
            component_groups=COMPONENT_GROUPS,
        )

    def write_module_data_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_module_rows_csv(
            results,
            path,
            component_groups=COMPONENT_GROUPS,
        )

    def write_architecture_metrics_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_architecture_metrics_rows_csv(results, path)

    def _run_network_variant(
        self,
        network: str,
        layers: Sequence[str],
        variant: MacroVariant,
    ) -> List[SimulationResult]:
        runs: List[TimeloopRun] = []
        for layer in layers:
            for architecture in self.spec.architecture_settings:
                macro_config = architecture.to_macro_config(
                    variant,
                    system=self.spec.system,
                )
                row_index = _row_index(variant.macro, network, layer, architecture)
                runs.append(
                    TimeloopRun(
                        layer=TimeloopLayerRef(network=network, layer_path=layer),
                        architecture=macro_config,
                        metadata={
                            "row_index": row_index,
                            "layer_id": _layer_id(layer),
                            "variant_label": variant.label,
                        },
                    )
                )
        return self.backend.run_batch(runs, n_jobs=self.spec.n_jobs)

    def _run_hybrid_spec(
        self, hybrid_spec: HybridMappingSpec, layers: Sequence[str]
    ) -> List[SimulationResult]:
        runs: List[TimeloopRun] = []
        for layer in layers:
            variant = hybrid_spec.variant_for_layer(_layer_id(layer))
            macro_config = hybrid_spec.architecture.to_macro_config(
                variant,
                system=self.spec.system,
            )
            row_index = _row_index(
                variant.macro,
                hybrid_spec.network,
                layer,
                hybrid_spec.architecture,
            )
            runs.append(
                TimeloopRun(
                    layer=TimeloopLayerRef(
                        network=hybrid_spec.network,
                        layer_path=layer,
                    ),
                    architecture=macro_config,
                    metadata={
                        "row_index": row_index,
                        "layer_id": _layer_id(layer),
                        "variant_label": variant.label,
                        "hybrid_family": hybrid_spec.family,
                    },
                )
            )
        return self.backend.run_batch(runs, n_jobs=self.spec.n_jobs)

    def _write_hybrid_aliases(
        self, hybrid_spec: HybridMappingSpec, paths: ArtifactPaths
    ) -> None:
        underscore_breakdown = (
            self.results_dir
            / f"results_{hybrid_spec.save_name}_{hybrid_spec.network}_breakdown_"
            f"{hybrid_spec.output_postfix}.csv"
        )
        if underscore_breakdown != paths.breakdown_csv:
            shutil.copyfile(paths.breakdown_csv, underscore_breakdown)

        if hybrid_spec.family == "delay-line":
            legacy_dir = self.results_dir / hybrid_spec.network
            legacy_dir.mkdir(parents=True, exist_ok=True)
            legacy_path = legacy_dir / "results_proposed_mrr_1bit_input_hybrid.csv"
            shutil.copyfile(paths.breakdown_csv, legacy_path)


def default_rosa_workflow(
    *,
    results_dir: Path = Path("results"),
    networks: Sequence[str] = DEFAULT_ROSA_NETWORKS,
    n_jobs: int = 128,
    backend: Optional[TimeloopBackend] = None,
) -> RosaWorkflow:
    variants = (
        MacroVariant(
            macro="proposed_mrr",
            label="MRR WS baseline",
            output_postfix="_1bit_input",
        ),
        MacroVariant(
            macro="proposed_mrr_optical_shift_add",
            label="MRR WS optical shift-add",
            output_postfix="_1bit_input_osa",
        ),
    )
    architectures = tuple(
        ArchitectureSetting(*raw_arch) for raw_arch in DEFAULT_ARCHITECTURES
    )
    spec = RosaWorkflowSpec(
        networks=tuple(networks),
        macro_variants=variants,
        architecture_settings=architectures,
        results_dir=results_dir,
        n_jobs=n_jobs,
    )
    return RosaWorkflow(spec, backend=backend)


def score_networks(
    *,
    networks: Tuple[str, ...],
    metrics_dir: Path,
    save_name: str,
    output_postfix: str,
    alpha: float,
    beta: float,
    lambda_worst_case: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.concat(
        [
            load_network_metrics(
                metrics_dir / f"architecture_metrics_{save_name}_{network}{output_postfix}.csv",
                network,
            )
            for network in networks
        ],
        ignore_index=True,
    )
    detailed = pd.concat(
        [
            score_network(metrics[metrics["Network"] == network], alpha=alpha, beta=beta)
            for network in networks
        ],
        ignore_index=True,
    )
    ranking = aggregate_scores(
        detailed,
        networks=networks,
        lambda_worst_case=lambda_worst_case,
    )
    return metrics, detailed, ranking


def load_network_metrics(path: Path, network: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    parsed = pd.DataFrame([_parse_architecture(value) for value in df["Architecture"]])
    df = pd.concat([df, parsed], axis=1)
    df.insert(0, "Network", network)
    return df


def score_network(metrics_df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    baseline_rows = metrics_df[(metrics_df["Cols"] == 4) & (metrics_df["Rows"] == 4)]
    if len(baseline_rows) != 1:
        network = metrics_df["Network"].iloc[0]
        raise ValueError(f"Expected one C4/R4 baseline for {network}, got {len(baseline_rows)}")
    baseline = baseline_rows.iloc[0]
    scored = metrics_df.copy()
    scored["Relative_Latency"] = scored["Latency"] / baseline["Latency"]
    scored["Relative_Energy_per_MAC"] = (
        scored["Energy_per_MAC"] / baseline["Energy_per_MAC"]
    )
    scored["Combined_Score"] = (
        scored["Relative_Latency"] ** alpha
        * scored["Relative_Energy_per_MAC"] ** beta
    )
    return scored


def aggregate_scores(
    scored_df: pd.DataFrame,
    *,
    networks: Tuple[str, ...],
    lambda_worst_case: float,
) -> pd.DataFrame:
    rows = []
    for architecture, group in scored_df.groupby("Architecture", sort=False):
        scores_by_network = group.set_index("Network")["Combined_Score"].reindex(networks)
        if scores_by_network.isna().any():
            missing = scores_by_network[scores_by_network.isna()].index.tolist()
            raise ValueError(f"Architecture {architecture} is missing scores for: {missing}")
        scores = scores_by_network.astype(float)
        geometric_mean = math.prod(scores) ** (1.0 / len(scores))
        worst_case_score = scores.max()
        best_network = scores.idxmin()
        worst_network = scores.idxmax()
        rows.append(
            {
                "Architecture": architecture,
                "Aggregated_Score": (
                    geometric_mean * (1.0 - lambda_worst_case)
                    + worst_case_score * lambda_worst_case
                ),
                "Geometric_Mean": geometric_mean,
                "Worst_Case_Score": worst_case_score,
                "Best_Network": best_network,
                "Best_Network_Score": scores[best_network],
                "Worst_Network": worst_network,
                "Worst_Network_Score": scores[worst_network],
            }
        )
    ranking = (
        pd.DataFrame(rows)
        .sort_values("Aggregated_Score", kind="mergesort")
        .reset_index(drop=True)
    )
    ranking.insert(2, "Rank", range(1, len(ranking) + 1))
    return ranking


def parse_architecture_argument(value: str) -> ArchitectureSetting:
    value = value.strip()
    if value.startswith("T"):
        parsed = parse_architecture_key(value)
        return ArchitectureSetting(
            parsed["tiles"], parsed["pes"], parsed["cols"], parsed["rows"]
        )
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("architecture must be T/P/C/R text or four comma-separated ints")
    return ArchitectureSetting(parts[0], parts[1], parts[2], parts[3])


def _parse_architecture(architecture: str) -> Dict[str, int]:
    match = ARCH_RE.fullmatch(str(architecture).strip().replace('"', ""))
    if not match:
        raise ValueError(f"Cannot parse architecture string: {architecture!r}")
    return {key: int(value) for key, value in match.groupdict().items()}


def _row_architecture(row: Mapping[str, object]) -> str:
    return f"T{int(row['Tiles'])},P{int(row['PEs'])},C{int(row['Cols'])},R{int(row['Rows'])}"


def _row_index(
    macro: str,
    network: str,
    layer: str,
    architecture: ArchitectureSetting,
) -> str:
    layer_file = f"workspace/models/workloads/{layer}.yaml"
    return f"{macro}_{network}_{layer_file}_{architecture.filename_key}"


def _layer_id(layer: str) -> str:
    return layer.rsplit("/", 1)[-1].replace(".yaml", "")


def _hybrid_family_macros(family: str) -> Tuple[str, str]:
    if family == "osa":
        return "proposed_mrr_optical_shift_add", "proposed_mrr_wi_optical_shift_add"
    if family == "delay-line":
        return (
            "proposed_mrr_1bit_input_delay_line",
            "proposed_mrr_1bit_input_delay_line_wi",
        )
    raise ValueError(f"Unknown hybrid family: {family!r}")


def _resolve_path(repo_root: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_workspace_dir(repo_root: Path) -> Path:
    package_workspace = Path(__file__).resolve().parents[2] / "workspace"
    if package_workspace.exists():
        return package_workspace
    return repo_root / "workspace"
