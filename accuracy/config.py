"""Validated configuration objects for accuracy experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class MRRVariationConfig:
    """Gaussian parameters consumed by ONNSim's uncalibrated MRR model.

    ``thermal_std`` is the effective standard deviation at
    ``thermal_reference_bits``. Lower-width slices use a reduced code-domain
    sensitivity; this does not imply that slicing changes physical temperature
    variance.
    """

    thermal_std: float = 0.05
    dac_std: float = 0.02
    thermal_scaling_exponent: float = 0.5
    thermal_reference_bits: int = 8

    def __post_init__(self) -> None:
        if self.thermal_std < 0 or self.dac_std < 0:
            raise ValueError("MRR variation standard deviations must be non-negative")
        if self.thermal_scaling_exponent < 0:
            raise ValueError("thermal scaling exponent must be non-negative")
        if self.thermal_reference_bits not in {1, 2, 4, 8}:
            raise ValueError("thermal reference bits must be one of 1, 2, 4, or 8")

    def effective_thermal_std(self, slice_bits: int) -> float:
        """Return effective thermal sensitivity for one supported slice width."""
        if slice_bits not in {1, 2, 4, 8}:
            raise ValueError("slice bits must be one of 1, 2, 4, or 8")
        level_ratio = (2**slice_bits - 1) / (2**self.thermal_reference_bits - 1)
        return self.thermal_std * level_ratio**self.thermal_scaling_exponent


@dataclass(frozen=True)
class AccuracyExperimentConfig:
    """Inputs needed for a reproducible whole-network accuracy evaluation."""

    network: str
    dataset: str
    checkpoint: Path
    model_config: Path
    runs: int = 5
    seeds: Tuple[int, ...] = field(default_factory=lambda: (0, 1, 2, 3, 4))
    variation: MRRVariationConfig = field(default_factory=MRRVariationConfig)

    def __post_init__(self) -> None:
        if not self.network or not self.dataset:
            raise ValueError("network and dataset cannot be empty")
        if self.runs <= 0:
            raise ValueError("runs must be positive")
        if len(self.seeds) != self.runs:
            raise ValueError("exactly one seed is required per run")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "model_config", Path(self.model_config))
