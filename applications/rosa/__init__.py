"""ROSA application workflow for OpticalLoop."""

from opticalloop.applications.rosa.validation import (
    FINAL_ARTIFACTS,
    RosaResultValidator,
    ValidationCheck,
    write_reference_artifacts,
)
from opticalloop.applications.rosa.workflow import (
    ArchitectureSetting,
    HybridMappingSpec,
    MacroVariant,
    RosaWorkflow,
    RosaWorkflowSpec,
    default_rosa_workflow,
    parse_architecture_argument,
)

__all__ = [
    "ArchitectureSetting",
    "FINAL_ARTIFACTS",
    "HybridMappingSpec",
    "MacroVariant",
    "RosaResultValidator",
    "RosaWorkflow",
    "RosaWorkflowSpec",
    "ValidationCheck",
    "default_rosa_workflow",
    "parse_architecture_argument",
    "write_reference_artifacts",
]
