"""DEAP-CNNs application workflow for OpticalLoop."""

from opticalloop.applications.deap_cnns.validation import (
    DeapResultValidator,
    ValidationCheck,
    write_deap_artifacts,
)
from opticalloop.applications.deap_cnns.workflow import (
    DEAP_ARCHITECTURES,
    DEAP_MACRO,
    DEAP_NETWORK,
    DeapArchitectureSetting,
    DeapDeviceSpec,
    DeapWorkflow,
    DeapWorkflowSpec,
    architecture_summary_dataframe,
    deap_architecture_by_name,
    default_deap_workflow,
    device_parameters_dataframe,
)

__all__ = [
    "DEAP_ARCHITECTURES",
    "DEAP_MACRO",
    "DEAP_NETWORK",
    "DeapArchitectureSetting",
    "DeapDeviceSpec",
    "DeapResultValidator",
    "DeapWorkflow",
    "DeapWorkflowSpec",
    "ValidationCheck",
    "architecture_summary_dataframe",
    "deap_architecture_by_name",
    "default_deap_workflow",
    "device_parameters_dataframe",
    "write_deap_artifacts",
]
