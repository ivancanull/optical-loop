"""Backend boundary for optional neural-network accuracy simulators."""

from abc import ABC, abstractmethod

from opticalloop.accuracy.config import AccuracyExperimentConfig
from opticalloop.accuracy.layer_manifest import LayerPolicy
from opticalloop.accuracy.result import AccuracyResult


class AccuracyBackend(ABC):
    """Run whole-network accuracy for one layer mapping policy."""

    @abstractmethod
    def run(
        self,
        experiment: AccuracyExperimentConfig,
        policy: LayerPolicy,
    ) -> AccuracyResult:
        """Evaluate a model and return aggregate and per-run accuracy."""
