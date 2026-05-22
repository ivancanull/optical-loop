"""Core model objects for Optical Loop."""

from opticalloop.model.mapping import MappingConfig
from opticalloop.model.optical_array import OpticalArray
from opticalloop.model.workload import LinearLayer

__all__ = ["LinearLayer", "MappingConfig", "OpticalArray"]
