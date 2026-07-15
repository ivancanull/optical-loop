"""Timeloop-backed Python frontend for OpticalLoop."""

from opticalloop.backend import TimeloopBackend, TimeloopRun
from opticalloop.cache import ArchitectureMetric, TimeloopResultCache
from opticalloop.config.architecture import MRRMacroConfig, TimeloopMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.module_data import ModuleSimulationData, module_dataframe
from opticalloop.result import SimulationResult
from opticalloop.simulator.layer_simulator import LayerSimulator

__all__ = [
    "ArchitectureMetric",
    "LayerSimulator",
    "MRRMacroConfig",
    "ModuleSimulationData",
    "SimulationResult",
    "TimeloopBackend",
    "TimeloopLayerRef",
    "TimeloopMacroConfig",
    "TimeloopResultCache",
    "TimeloopRun",
    "module_dataframe",
]
