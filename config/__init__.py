"""Configuration dataclasses for OpticalLoop."""

from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import LinearLayerConfig, TimeloopLayerRef

__all__ = ["LinearLayerConfig", "MRRMacroConfig", "TimeloopLayerRef"]
