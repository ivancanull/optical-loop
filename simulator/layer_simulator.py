"""Timeloop-backed layer simulator."""

from dataclasses import dataclass
from typing import Optional

from opticalloop.backend import TimeloopBackend
from opticalloop.cache import TimeloopResultCache
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.result import SimulationResult


@dataclass(frozen=True)
class LayerSimulator:
    """Return Timeloop/CIMLoop result data for one workload layer."""

    layer: TimeloopLayerRef
    architecture: MRRMacroConfig
    backend: Optional[TimeloopBackend] = None
    cache: Optional[TimeloopResultCache] = None
    prefer_cache: bool = True

    def run(self) -> SimulationResult:
        if self.prefer_cache and self.cache is not None:
            cached = self.cache.get_layer_result(self.layer, self.architecture)
            if cached is not None:
                return cached

        backend = self.backend
        if backend is None:
            if self.cache is not None:
                raise FileNotFoundError(
                    "No cached Timeloop result matched the requested layer and architecture"
                )
            backend = TimeloopBackend()

        return backend.run_layer(self.layer, self.architecture)
