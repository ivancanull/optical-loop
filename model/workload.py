"""Workload model wrappers."""

from dataclasses import dataclass

from opticalloop.config.workload import LinearLayerConfig


@dataclass(frozen=True)
class LinearLayer:
    """Runtime workload model for a dense linear layer."""

    config: LinearLayerConfig

    @property
    def macs(self) -> int:
        return self.config.macs
