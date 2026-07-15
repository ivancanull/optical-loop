"""Paper-facing EDP reproduction from committed Timeloop aggregates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


PAPER_NETWORKS = (
    "alexnet",
    "vgg16",
    "resnet18",
    "mobilenet_v3",
    "gpt2_medium",
    "vision_transformer",
)


@dataclass(frozen=True)
class PaperEDPConfig:
    """Inputs used by the architecture-level DAC26 EDP comparison."""

    data_dir: Path
    networks: Sequence[str] = PAPER_NETWORKS
    optimized_cols: int = 8
    optimized_rows: int = 8
    compact_cols: int = 4
    compact_rows: int = 4
    deap_cols: int = 9
    deap_rows: int = 113


class PaperEDPReproduction:
    """Recompute paper comparisons without any accuracy-model dependency."""

    def __init__(self, config: PaperEDPConfig) -> None:
        self.config = config

    def load_metrics(self) -> pd.DataFrame:
        frames = []
        for network in self.config.networks:
            for osa, suffix in ((False, ""), (True, "_osa")):
                path = self.config.data_dir / (
                    f"aggregated_metrics_{network}_1bit_input{suffix}.csv"
                )
                frame = pd.read_csv(path)
                frame.insert(0, "Network", network)
                frame.insert(1, "OSA", osa)
                frames.append(frame)
        metrics = pd.concat(frames, ignore_index=True)
        # Timeloop exports EDP, but recompute it here to make the unit boundary explicit.
        metrics["Recomputed_EDP"] = metrics["Energy"] * metrics["Latency"]
        return metrics

    def architecture_summary(self, *, osa: bool = False) -> pd.DataFrame:
        metrics = self.load_metrics()
        metrics = metrics[metrics["OSA"] == osa]
        rows = []
        for (cols, rows_count), group in metrics.groupby(["Cols", "Rows"], sort=False):
            if set(group["Network"]) != set(self.config.networks):
                continue
            rows.append(
                {
                    "Cols": int(cols),
                    "Rows": int(rows_count),
                    "Geometric_Mean_EDP": _geometric_mean(group["Recomputed_EDP"]),
                }
            )
        return pd.DataFrame(rows).sort_values("Geometric_Mean_EDP").reset_index(drop=True)

    def headline_summary(self) -> dict[str, float | str]:
        no_osa = self.architecture_summary(osa=False)
        # The two DEAP shapes are comparison points, not candidates: DAC26 caps
        # the optimized design space at eight wavelength columns.
        eligible = no_osa[no_osa["Cols"] <= 8]
        best = eligible.iloc[0]
        optimized = self._architecture_edp(no_osa, self.config.optimized_cols, self.config.optimized_rows)
        compact = self._architecture_edp(no_osa, self.config.compact_cols, self.config.compact_rows)
        deap = self._architecture_edp(no_osa, self.config.deap_cols, self.config.deap_rows)

        osa = self.architecture_summary(osa=True)
        optimized_osa = self._architecture_edp(
            osa, self.config.optimized_cols, self.config.optimized_rows
        )
        return {
            "best_no_osa": f"C{int(best['Cols'])}xR{int(best['Rows'])}",
            "optimized_vs_compact_reduction": 1.0 - optimized / compact,
            "optimized_vs_deap_reduction": 1.0 - optimized / deap,
            "osa_reduction_at_optimized_shape": 1.0 - optimized_osa / optimized,
        }

    def formula_checks(self, tolerance: float = 1e-9) -> pd.DataFrame:
        metrics = self.load_metrics()
        relative_error = (
            (metrics["EDP"] - metrics["Recomputed_EDP"]).abs()
            / metrics["EDP"].abs().clip(lower=1e-30)
        )
        return pd.DataFrame(
            [
                {
                    "check": "all_rows_edp_equals_energy_times_latency",
                    "expected": True,
                    "actual": bool((relative_error <= tolerance).all()),
                    "max_relative_error": float(relative_error.max()),
                    "passed": bool((relative_error <= tolerance).all()),
                },
                {
                    "check": "no_osa_geometric_mean_winner",
                    "expected": "C8xR8",
                    "actual": self.headline_summary()["best_no_osa"],
                    "max_relative_error": 0.0,
                    "passed": self.headline_summary()["best_no_osa"] == "C8xR8",
                },
            ]
        )

    @staticmethod
    def _architecture_edp(summary: pd.DataFrame, cols: int, rows: int) -> float:
        selected = summary[(summary["Cols"] == cols) & (summary["Rows"] == rows)]
        if len(selected) != 1:
            raise ValueError(f"Expected one C{cols}/R{rows} result, found {len(selected)}")
        return float(selected.iloc[0]["Geometric_Mean_EDP"])


def _geometric_mean(values: pd.Series) -> float:
    values = values.astype(float)
    if (values <= 0).any():
        raise ValueError("EDP values must be positive for a geometric mean")
    return math.exp(sum(math.log(value) for value in values) / len(values))
