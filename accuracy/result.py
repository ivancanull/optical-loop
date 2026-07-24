"""Results returned by accuracy backends."""

from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class AccuracyResult:
    """Whole-network accuracy distribution for one mapping policy."""

    network: str
    scenario: str
    accuracies: Tuple[float, ...]
    losses: Tuple[float, ...]
    seeds: Tuple[int, ...]
    baseline_accuracy: Optional[float] = None
    source: str = "accuracy-backend"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.accuracies)
        if count == 0 or len(self.losses) != count or len(self.seeds) != count:
            raise ValueError("accuracies, losses, and seeds must have equal non-zero length")
        if any(value < 0 or value > 100 for value in self.accuracies):
            raise ValueError("accuracy values must be percentages in [0, 100]")

    @property
    def accuracy_mean(self) -> float:
        return fmean(self.accuracies)

    @property
    def accuracy_std(self) -> float:
        return stdev(self.accuracies) if len(self.accuracies) > 1 else 0.0

    @property
    def loss_mean(self) -> float:
        return fmean(self.losses)

    @property
    def accuracy_delta(self) -> Optional[float]:
        if self.baseline_accuracy is None:
            return None
        return self.accuracy_mean - self.baseline_accuracy

    def to_row(self) -> dict[str, object]:
        return {
            "network": self.network,
            "scenario": self.scenario,
            "accuracy": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "accuracy_delta": self.accuracy_delta,
            "loss": self.loss_mean,
            "accuracy_runs": len(self.accuracies),
            "accuracy_source": self.source,
        }
