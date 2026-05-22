"""Workload metadata wrappers."""

from dataclasses import dataclass

from opticalloop.config.workload import LinearLayerConfig


@dataclass(frozen=True)
class LinearLayer:
    """Metadata wrapper for a dense linear layer."""

    config: LinearLayerConfig

    @property
    def macs(self) -> int:
        return self.config.macs
