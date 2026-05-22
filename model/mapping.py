"""Mapping configuration and access-count boundary for Timeloop adapters."""

from dataclasses import dataclass
from typing import Mapping, Optional


_ACCESS_KEYS = (
    "input_dac",
    "weight_dac",
    "input_mrr",
    "weight_mrr",
    "photodiode",
    "tia",
    "adc",
)


def _require_positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True)
class MappingConfig:
    """Loop mapping/reuse summary for the minimal simulator.

    Reuse factors divide the default analytic access counts. Explicit access
    counts override only the components provided, which is the intended V1
    adapter boundary for later Timeloop integration.
    """

    input_reuse: float = 1.0
    weight_reuse: float = 1.0
    output_reuse: float = 1.0
    access_counts: Optional[Mapping[str, int]] = None

    def __post_init__(self) -> None:
        _require_positive("input_reuse", self.input_reuse)
        _require_positive("weight_reuse", self.weight_reuse)
        _require_positive("output_reuse", self.output_reuse)
        if self.access_counts is None:
            return
        unknown = sorted(set(self.access_counts) - set(_ACCESS_KEYS))
        if unknown:
            raise ValueError(f"unknown access count keys: {unknown}")
        for key, value in self.access_counts.items():
            _require_non_negative_int(f"access_counts[{key}]", value)

    @classmethod
    def from_access_counts(
        cls,
        access_counts: Mapping[str, int],
        *,
        input_reuse: float = 1.0,
        weight_reuse: float = 1.0,
        output_reuse: float = 1.0,
    ) -> "MappingConfig":
        """Create a mapping from externally supplied component access counts."""

        return cls(
            input_reuse=input_reuse,
            weight_reuse=weight_reuse,
            output_reuse=output_reuse,
            access_counts=dict(access_counts),
        )

    def override_for(self, component: str) -> Optional[int]:
        if self.access_counts is None:
            return None
        return self.access_counts.get(component)
