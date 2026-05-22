"""Timeloop-backed Python frontend for OpticalLoop."""

from opticalloop.backend import TimeloopBackend, TimeloopRun
from opticalloop.cache import ArchitectureMetric, TimeloopResultCache
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.module_data import ModuleSimulationData, module_dataframe
from opticalloop.result import SimulationResult
from opticalloop.simulator.layer_simulator import LayerSimulator
from opticalloop.workflow import (
    RosaResultValidator,
    RosaWorkflow,
    RosaWorkflowSpec,
    default_rosa_workflow,
    write_reference_artifacts,
)

__all__ = [
    "ArchitectureMetric",
    "LayerSimulator",
    "MRRMacroConfig",
    "ModuleSimulationData",
    "SimulationResult",
    "TimeloopBackend",
    "TimeloopLayerRef",
    "TimeloopResultCache",
    "TimeloopRun",
    "RosaWorkflow",
    "RosaWorkflowSpec",
    "RosaResultValidator",
    "default_rosa_workflow",
    "module_dataframe",
    "write_reference_artifacts",
]
