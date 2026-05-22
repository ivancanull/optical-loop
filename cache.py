"""Readers for persisted Timeloop result CSVs."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.result import SimulationResult


_ARCH_RE = re.compile(r"T(?P<tiles>\d+),\s*P(?P<pes>\d+),\s*C(?P<cols>\d+),\s*R(?P<rows>\d+)")


@dataclass(frozen=True)
class ArchitectureMetric:
    """Architecture-level metric row read from a Timeloop-generated CSV."""

    architecture: str
    cycles: int
    latency_s: float
    energy_j: Optional[float]
    area_mm2: Optional[float]
    edp: float
    tops: Optional[float]
    tops_per_w: Optional[float]
    tops_per_mm2: Optional[float]


@dataclass(frozen=True)
class TimeloopResultCache:
    """Read Timeloop-generated result CSVs without estimating new values."""

    results_dir: Path = Path("results")
    save_name: str = "deapcnns"
    output_postfix: str = "_1bit_input_osa"

    def get_layer_result(
        self, layer: TimeloopLayerRef, architecture: MRRMacroConfig
    ) -> Optional[SimulationResult]:
        path = (
            self.results_dir
            / "reconstructed"
            / f"results_{layer.network}_breakdown{self.output_postfix}.csv"
        )
        if not path.exists():
            return None

        for row in self._read_dicts(path):
            if not self._matches_layer_row(row, layer, architecture):
                continue
            return self._result_from_layer_row(row, path, layer, architecture)
        return None

    def read_architecture_metrics(self, network: str) -> List[ArchitectureMetric]:
        path = self.results_dir / f"architecture_metrics_{self.save_name}_{network}{self.output_postfix}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        metrics = []
        for row in self._read_dicts(path):
            energy_j = None
            if row.get("Energy"):
                energy_j = self._float(row["Energy"])
            metrics.append(
                ArchitectureMetric(
                    architecture=row["Architecture"],
                    cycles=self._int(row["Total_Cycles"]),
                    latency_s=self._float(row["Latency"]),
                    energy_j=energy_j,
                    area_mm2=self._optional_float(row.get("Total_Area")),
                    edp=self._float(row["EDP"]),
                    tops=self._optional_float(row.get("TOPS")),
                    tops_per_w=self._optional_float(row.get("TOPS_per_W")),
                    tops_per_mm2=self._optional_float(row.get("TOPS_per_mm2")),
                )
            )
        return metrics

    def get_architecture_metric(
        self, network: str, architecture: MRRMacroConfig
    ) -> Optional[ArchitectureMetric]:
        for metric in self.read_architecture_metrics(network):
            parsed = parse_architecture_key(metric.architecture)
            if parsed == {
                "tiles": architecture.n_tiles,
                "pes": architecture.n_pes,
                "cols": architecture.n_cols,
                "rows": architecture.n_rows,
            }:
                return metric
        return None

    @staticmethod
    def _read_dicts(path: Path) -> List[Dict[str, str]]:
        with path.open(newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    @staticmethod
    def _matches_layer_row(
        row: Dict[str, str], layer: TimeloopLayerRef, architecture: MRRMacroConfig
    ) -> bool:
        return (
            str(row.get("Network")) == layer.network
            and str(row.get("Layer_Path")) == layer.layer_id
            and int(row.get("Tiles", -1)) == architecture.n_tiles
            and int(row.get("PEs", -1)) == architecture.n_pes
            and int(row.get("Cols", -1)) == architecture.n_cols
            and int(row.get("Rows", -1)) == architecture.n_rows
        )

    @classmethod
    def _result_from_layer_row(
        cls,
        row: Dict[str, str],
        path: Path,
        layer: TimeloopLayerRef,
        architecture: MRRMacroConfig,
    ) -> SimulationResult:
        return SimulationResult(
            compute=cls._optional_float(row.get("compute")),
            cycles=cls._int(row["cycles"]),
            cycle_seconds=cls._optional_float(row.get("cycle_seconds")),
            latency_s=cls._float(row["latency"]),
            energy_j=cls._float(row["energy"]),
            area_mm2=cls._optional_float(row.get("area")),
            tops=cls._optional_float(row.get("tops")),
            tops_per_w=cls._optional_float(row.get("tops_per_w")),
            tops_per_mm2=cls._optional_float(row.get("tops_per_mm2")),
            energy_breakdown=component_energy_from_row(row),
            area_breakdown=component_metric_from_row(row, "area"),
            power_breakdown=component_metric_from_row(row, "power"),
            source="timeloop-cache",
            metadata={
                "path": str(path),
                "network": layer.network,
                "layer_path": layer.layer_path,
                "architecture": architecture.architecture_key,
                "macro": architecture.macro,
            },
        )

    @staticmethod
    def _float(value: str) -> float:
        return float(value)

    @staticmethod
    def _optional_float(value: Optional[str]) -> Optional[float]:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _int(value: str) -> int:
        return int(float(value))


def parse_architecture_key(value: str) -> Dict[str, int]:
    match = _ARCH_RE.fullmatch(str(value).strip().replace('"', ""))
    if not match:
        raise ValueError(f"Cannot parse architecture string: {value!r}")
    return {key: int(raw) for key, raw in match.groupdict().items()}


def component_energy_from_row(row: Dict[str, str]) -> Dict[str, float]:
    return component_metric_from_row(row, "energy")


def component_metric_from_row(row: Dict[str, str], metric: str) -> Dict[str, float]:
    breakdown = {}
    suffix = f"_{metric}"
    for key, value in row.items():
        if key == metric or not key.endswith(suffix) or value in (None, ""):
            continue
        breakdown[key[: -len(suffix)]] = float(value)
    return breakdown
