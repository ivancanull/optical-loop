"""Configuration dataclasses for Optical Loop."""

from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import LinearLayerConfig, TimeloopLayerRef

__all__ = ["LinearLayerConfig", "MRRMacroConfig", "TimeloopLayerRef"]
