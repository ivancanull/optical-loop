"""Timeloop-backed optical macro architecture configuration objects."""

from dataclasses import dataclass, field
from typing import Mapping, Optional


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_positive_float(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class TimeloopMacroConfig:
    """Generic Timeloop macro configuration.

    This is the lowest-level public architecture object: it forwards a macro
    name and an explicit variable dictionary to Timeloop without assuming an
    MRR row/column naming convention.
    """

    macro: str
    variables: Mapping[str, object] = field(default_factory=dict)
    system: str = "fetch_all_lpddr4"
    architecture_key: Optional[str] = None
    max_utilization: bool = True

    def __post_init__(self) -> None:
        if not self.macro:
            raise ValueError("macro cannot be empty")
        if not self.system:
            raise ValueError("system cannot be empty")
        object.__setattr__(self, "variables", dict(self.variables))
        if self.architecture_key is None:
            object.__setattr__(self, "architecture_key", self._default_key())

    def to_timeloop_variables(self) -> dict:
        return dict(self.variables)

    def _default_key(self) -> str:
        if not self.variables:
            return self.macro
        variable_text = ", ".join(
            f"{key}={self.variables[key]}" for key in sorted(self.variables)
        )
        return f"{self.macro}({variable_text})"


@dataclass(frozen=True)
class MRRMacroConfig:
    """MRR macro settings forwarded to Timeloop.

    The shape fields map directly onto Timeloop variables. Timing, energy, and
    area are intentionally not modeled here because they must come from
    Timeloop output statistics.
    """

    n_tiles: int
    n_pes: int
    n_rows: int
    n_cols: int
    macro: str = "proposed_mrr_optical_shift_add"
    system: str = "fetch_all_lpddr4"
    voltage_dac_resolution: int = 1
    scaling: str = '"aggressive"'
    max_utilization: bool = True
    frequency_hz: Optional[float] = None
    area_mm2: Optional[float] = None

    def __post_init__(self) -> None:
        _require_positive_int("n_tiles", self.n_tiles)
        _require_positive_int("n_pes", self.n_pes)
        _require_positive_int("n_rows", self.n_rows)
        _require_positive_int("n_cols", self.n_cols)
        if not self.macro:
            raise ValueError("macro cannot be empty")
        if not self.system:
            raise ValueError("system cannot be empty")
        _require_positive_int("voltage_dac_resolution", self.voltage_dac_resolution)
        if not self.scaling:
            raise ValueError("scaling cannot be empty")
        if self.frequency_hz is not None:
            _require_positive_float("frequency_hz", self.frequency_hz)
        if self.area_mm2 is not None:
            _require_positive_float("area_mm2", self.area_mm2)

    @property
    def parallel_macs(self) -> int:
        return self.n_tiles * self.n_pes * self.n_rows * self.n_cols

    @property
    def architecture_key(self) -> str:
        return f"T{self.n_tiles}, P{self.n_pes}, C{self.n_cols}, R{self.n_rows}"

    def to_timeloop_variables(self) -> dict:
        return {
            "SCALING": self.scaling,
            "N_TILES": self.n_tiles,
            "N_PES": self.n_pes,
            "N_COLUMNS": self.n_cols,
            "N_ROWS": self.n_rows,
            "VOLTAGE_DAC_RESOLUTION": self.voltage_dac_resolution,
        }
