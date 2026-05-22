"""Reusable Timeloop result CSV processing helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from opticalloop.result import SimulationResult


@dataclass(frozen=True)
class ArtifactPaths:
    """Standard result paths for one network and output variant."""

    breakdown_csv: Path
    combined_csv: Path
    module_data_csv: Path
    metrics_csv: Path
    reconstructed_breakdown_csv: Path
    aggregated_metrics_csv: Path


def artifact_paths(
    save_name: str,
    network: str,
    output_postfix: str,
    results_dir: Path,
) -> ArtifactPaths:
    reconstructed_dir = Path(results_dir) / "reconstructed"
    return ArtifactPaths(
        breakdown_csv=Path(results_dir)
        / f"results_{save_name}_{network}_breakdown{output_postfix}.csv",
        combined_csv=Path(results_dir)
        / f"results_{save_name}_{network}_combined{output_postfix}.csv",
        module_data_csv=Path(results_dir)
        / f"module_data_{save_name}_{network}{output_postfix}.csv",
        metrics_csv=Path(results_dir)
        / f"architecture_metrics_{save_name}_{network}{output_postfix}.csv",
        reconstructed_breakdown_csv=reconstructed_dir
        / f"results_{network}_breakdown{output_postfix}.csv",
        aggregated_metrics_csv=reconstructed_dir
        / f"aggregated_metrics_{network}{output_postfix}.csv",
    )


def write_results_csv(
    results: Sequence[SimulationResult],
    path: Path,
    *,
    group_components: bool,
    component_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _result_row(
            result,
            group_components=group_components,
            component_groups=component_groups or {},
        )
        for result in results
    ]
    columns = _ordered_columns(rows)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[""] + columns)
        writer.writeheader()
        for result, row in zip(results, rows):
            writer.writerow({"": result.metadata["row_index"], **row})
    return path


def write_architecture_metrics_csv(
    results: Sequence[SimulationResult], path: Path
) -> Path:
    rows = []
    for architecture, group in _group_by_architecture(results).items():
        compute = sum(result.compute or 0.0 for result in group)
        energy = sum(result.energy_j for result in group)
        cycles = sum(result.cycles for result in group)
        latency = sum(result.latency_s for result in group)
        area = next((result.area_mm2 for result in group if result.area_mm2), 0.0)
        cycle_seconds = next(
            (result.cycle_seconds for result in group if result.cycle_seconds), None
        )
        tops = sum(result.tops or 0.0 for result in group)
        power = energy / cycle_seconds / cycles if cycle_seconds and cycles else 0.0
        rows.append(
            {
                "Architecture": architecture,
                "TOPS": tops,
                "Energy_per_MAC": energy / compute if compute else 0.0,
                "Total_Area": area,
                "Total_Cycles": cycles,
                "Latency": latency,
                "Total_Power_W": power,
                "TOPS_per_W": tops / power if power else 0.0,
                "TOPS_per_mm2": tops / area if area else 0.0,
                "EDP": energy * latency,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def parse_detailed_data(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file, index_col=0)
    extracted_data = []
    for index, data_row in df.iterrows():
        parsed = _parse_breakdown_index(str(index))
        row_dict = data_row.to_dict()
        row_dict.update(parsed)
        extracted_data.append(row_dict)
    return pd.DataFrame(extracted_data)


def calculate_aggregated_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    architectures = dataframe[["Tiles", "PEs", "Cols", "Rows"]].drop_duplicates()
    aggregated_metrics = []
    for _, arch in architectures.iterrows():
        tile, pe, col, row = arch["Tiles"], arch["PEs"], arch["Cols"], arch["Rows"]
        filtered_df = dataframe[
            (dataframe["Tiles"] == tile)
            & (dataframe["PEs"] == pe)
            & (dataframe["Cols"] == col)
            & (dataframe["Rows"] == row)
        ]
        total_compute = filtered_df["compute"].sum()
        total_energy = filtered_df["energy"].sum()
        total_area = filtered_df["area"].iloc[0]
        total_cycles = filtered_df["cycles"].sum()
        total_latency = filtered_df["latency"].sum()
        cycle_seconds = filtered_df["cycle_seconds"].iloc[0]
        total_power = (
            total_energy / cycle_seconds / total_cycles
            if cycle_seconds and total_cycles
            else 0.0
        )
        tops = filtered_df["tops"].sum()
        aggregated_metrics.append(
            {
                "Tiles": tile,
                "PEs": pe,
                "Cols": col,
                "Rows": row,
                "EDP": total_energy * total_latency,
                "TOPS": tops,
                "Energy": total_energy,
                "Area": total_area,
                "Cycles": total_cycles,
                "Latency": total_latency,
                "Power": total_power,
                "Energy_per_MAC": total_energy / total_compute if total_compute else 0.0,
                "TOPS_per_W": tops / total_power if total_power else 0.0,
                "TOPS_per_mm2": tops / total_area if total_area else 0.0,
            }
        )
    return pd.DataFrame(aggregated_metrics)


def reconstruct_breakdown_csv(breakdown_csv: Path, reconstructed_csv: Path) -> Path:
    df = parse_detailed_data(Path(breakdown_csv))
    reconstructed_csv = Path(reconstructed_csv)
    reconstructed_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(reconstructed_csv)
    return reconstructed_csv


def aggregate_metrics_csv(reconstructed_csv: Path, aggregated_csv: Path) -> Path:
    df = pd.read_csv(reconstructed_csv, index_col=0)
    agg_df = calculate_aggregated_metrics(df)
    aggregated_csv = Path(aggregated_csv)
    aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(aggregated_csv, index=False)
    return aggregated_csv


def _result_row(
    result: SimulationResult,
    *,
    group_components: bool,
    component_groups: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    row = {
        "compute": result.compute,
        "cycles": result.cycles,
        "cycle_seconds": result.cycle_seconds,
        "latency": result.latency_s,
        "energy": result.energy_j,
        "area": result.area_mm2,
        "tops": result.tops,
        "tops_per_mm2": result.tops_per_mm2,
        "tops_per_w": result.tops_per_w,
        "tops_per_mm2_w": _tops_per_mm2_w(result),
    }
    if group_components:
        component_names = _grouped_component_values(result, component_groups)
    else:
        component_names = set(result.energy_breakdown) | set(result.area_breakdown)

    for component in sorted(component_names):
        row[f"{component}_energy"] = _component_value(
            result, component, "energy", group_components, component_groups
        )
        row[f"{component}_area"] = _component_value(
            result, component, "area", group_components, component_groups
        )
        row[f"{component}_power"] = _component_value(
            result, component, "power", group_components, component_groups
        )
    return row


def _component_value(
    result: SimulationResult,
    component: str,
    kind: str,
    grouped: bool,
    component_groups: Mapping[str, Sequence[str]],
) -> float:
    source = {
        "energy": result.energy_breakdown,
        "area": result.area_breakdown,
        "power": result.power_breakdown,
    }[kind]
    if not grouped or component not in component_groups:
        return float(source.get(component, 0.0))
    return sum(float(source.get(name, 0.0)) for name in component_groups[component])


def _grouped_component_values(
    result: SimulationResult, component_groups: Mapping[str, Sequence[str]]
) -> Tuple[str, ...]:
    raw_components = set(result.energy_breakdown) | set(result.area_breakdown)
    grouped_sources = {name for names in component_groups.values() for name in names}
    grouped = [
        name for name, names in component_groups.items() if raw_components & set(names)
    ]
    leftovers = sorted(raw_components - grouped_sources)
    return tuple(grouped + leftovers)


def _ordered_columns(rows: Sequence[Mapping[str, object]]) -> List[str]:
    base = [
        "compute",
        "cycles",
        "cycle_seconds",
        "latency",
        "energy",
        "area",
        "tops",
        "tops_per_mm2",
        "tops_per_w",
        "tops_per_mm2_w",
    ]
    extras = sorted({key for row in rows for key in row if key not in base})
    return base + extras


def _tops_per_mm2_w(result: SimulationResult) -> Optional[float]:
    if result.tops_per_w is None or not result.area_mm2:
        return None
    return result.tops_per_w / result.area_mm2 / 1e6


def _group_by_architecture(
    results: Sequence[SimulationResult],
) -> Dict[str, List[SimulationResult]]:
    grouped: Dict[str, List[SimulationResult]] = {}
    for result in results:
        architecture = str(result.metadata["architecture"])
        grouped.setdefault(architecture, []).append(result)
    return grouped


def _parse_breakdown_index(index: str) -> Dict[str, object]:
    prefix, layer_part = index.rsplit("/", 1)
    network = Path(prefix).name
    parts = layer_part.split("_")
    layer = parts[0].replace(".yaml", "")
    return {
        "Original_Index": index,
        "Network": network,
        "Layer_Path": layer,
        "Tiles": int(parts[-4].replace("tiles", "")),
        "PEs": int(parts[-3].replace("pes", "")),
        "Cols": int(parts[-2].replace("cols", "")),
        "Rows": int(parts[-1].replace("rows", "")),
    }
