"""Result objects populated from Timeloop outputs."""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class SimulationResult:
    """Top-level result for one Timeloop-backed layer simulation."""

    cycles: int
    latency_s: float
    energy_j: float
    energy_breakdown: Dict[str, float]
    area_breakdown: Dict[str, float] = field(default_factory=dict)
    power_breakdown: Dict[str, float] = field(default_factory=dict)
    area_mm2: Optional[float] = None
    tops: Optional[float] = None
    tops_per_w: Optional[float] = None
    tops_per_mm2: Optional[float] = None
    compute: Optional[float] = None
    cycle_seconds: Optional[float] = None
    source: str = "timeloop"
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cycles < 0:
            raise ValueError(f"cycles must be non-negative, got {self.cycles!r}")
        if self.latency_s < 0:
            raise ValueError(f"latency_s must be non-negative, got {self.latency_s!r}")
        if self.energy_j < 0:
            raise ValueError(f"energy_j must be non-negative, got {self.energy_j!r}")
        if self.area_mm2 is not None and self.area_mm2 < 0:
            raise ValueError(f"area_mm2 must be non-negative, got {self.area_mm2!r}")

    @classmethod
    def from_timeloop_stats(
        cls,
        stats,
        *,
        source: str = "timeloop-live",
        metadata: Optional[Dict[str, object]] = None,
    ) -> "SimulationResult":
        cycles = int(getattr(stats, "cycles"))
        cycle_seconds = getattr(stats, "cycle_seconds", None)
        latency = getattr(stats, "latency", None)
        if latency is None:
            latency = cycles * cycle_seconds

        energy_breakdown = _float_mapping(
            getattr(stats, "per_component_energy", {}) or {}
        )
        area_breakdown = _float_mapping(getattr(stats, "per_component_area", {}) or {})
        power_breakdown = _float_mapping(
            getattr(stats, "per_component_power", {}) or {}
        )
        if not power_breakdown and cycle_seconds is not None and cycles:
            power_breakdown = {
                name: value / float(cycle_seconds) / cycles
                for name, value in energy_breakdown.items()
            }

        return cls(
            cycles=cycles,
            cycle_seconds=cycle_seconds,
            latency_s=float(latency),
            energy_j=float(getattr(stats, "energy")),
            area_mm2=(
                float(getattr(stats, "area"))
                if getattr(stats, "area", None) is not None
                else None
            ),
            tops=float(getattr(stats, "tops")) if getattr(stats, "tops", None) is not None else None,
            tops_per_w=(
                float(getattr(stats, "tops_per_w"))
                if getattr(stats, "tops_per_w", None) is not None
                else None
            ),
            tops_per_mm2=(
                float(getattr(stats, "tops_per_mm2"))
                if getattr(stats, "tops_per_mm2", None) is not None
                else None
            ),
            compute=(
                float(getattr(stats, "computes"))
                if getattr(stats, "computes", None) is not None
                else None
            ),
            energy_breakdown=energy_breakdown,
            area_breakdown=area_breakdown,
            power_breakdown=power_breakdown,
            source=source,
            metadata=dict(metadata or {}),
        )


def _float_mapping(values: Mapping[object, object]) -> Dict[str, float]:
    return {str(name): float(value) for name, value in values.items()}
