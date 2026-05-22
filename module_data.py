"""Tidy per-module simulation data derived from Timeloop results."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from opticalloop.result import SimulationResult


@dataclass(frozen=True)
class ModuleSimulationData:
    """One row of module-level Timeloop data.

    The row is intentionally long-form so it can be filtered, grouped, plotted,
    or joined without parsing component-specific wide CSV columns.
    """

    network: str
    layer_path: str
    layer_id: str
    macro: str
    architecture: str
    module: str
    module_group: str
    energy_j: float
    area_mm2: Optional[float]
    power_w: Optional[float]
    cycles: int
    latency_s: float
    total_energy_j: float
    total_area_mm2: Optional[float]
    source: str
    row_index: str


def module_rows_from_result(
    result: SimulationResult,
    *,
    component_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> Sequence[ModuleSimulationData]:
    """Convert one Timeloop result into tidy module rows."""

    groups_by_component = _groups_by_component(component_groups or {})
    modules = sorted(
        set(result.energy_breakdown)
        | set(result.area_breakdown)
        | set(result.power_breakdown)
    )
    return tuple(
        ModuleSimulationData(
            network=str(result.metadata.get("network", "")),
            layer_path=str(result.metadata.get("layer_path", "")),
            layer_id=str(
                result.metadata.get(
                    "layer_id",
                    str(result.metadata.get("layer_path", "")).rsplit("/", 1)[-1].replace(
                        ".yaml", ""
                    ),
                )
            ),
            macro=str(result.metadata.get("macro", "")),
            architecture=str(result.metadata.get("architecture", "")),
            module=module,
            module_group=groups_by_component.get(module, ""),
            energy_j=float(result.energy_breakdown.get(module, 0.0)),
            area_mm2=_optional_float(result.area_breakdown.get(module)),
            power_w=_module_power(result, module),
            cycles=result.cycles,
            latency_s=result.latency_s,
            total_energy_j=result.energy_j,
            total_area_mm2=result.area_mm2,
            source=result.source,
            row_index=str(result.metadata.get("row_index", "")),
        )
        for module in modules
    )


def module_dataframe(
    results: Iterable[SimulationResult],
    *,
    component_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Build a tidy DataFrame with one row per result module."""

    rows = [
        asdict(row)
        for result in results
        for row in module_rows_from_result(
            result, component_groups=component_groups
        )
    ]
    return pd.DataFrame(rows, columns=_columns())


def write_module_data_csv(
    results: Iterable[SimulationResult],
    path: Path,
    *,
    component_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> Path:
    """Write tidy module-level Timeloop data to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module_dataframe(results, component_groups=component_groups).to_csv(path, index=False)
    return path


def _groups_by_component(
    component_groups: Mapping[str, Sequence[str]]
) -> Dict[str, str]:
    grouped: Dict[str, str] = {}
    for group, components in component_groups.items():
        for component in components:
            grouped[str(component)] = str(group)
    return grouped


def _module_power(result: SimulationResult, module: str) -> Optional[float]:
    if module in result.power_breakdown:
        return float(result.power_breakdown[module])
    energy = result.energy_breakdown.get(module)
    if energy is None or result.cycle_seconds is None or result.cycles == 0:
        return None
    return float(energy) / float(result.cycle_seconds) / result.cycles


def _optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _columns() -> Sequence[str]:
    return (
        "network",
        "layer_path",
        "layer_id",
        "macro",
        "architecture",
        "module",
        "module_group",
        "energy_j",
        "area_mm2",
        "power_w",
        "cycles",
        "latency_s",
        "total_energy_j",
        "total_area_mm2",
        "source",
        "row_index",
    )
