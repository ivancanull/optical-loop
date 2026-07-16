"""ROSA application workflow for OpticalLoop."""

from opticalloop.applications.rosa.validation import (
    FINAL_ARTIFACTS,
    RosaResultValidator,
    ValidationCheck,
    write_reference_artifacts,
)
from opticalloop.applications.rosa.paper_edp import (
    PAPER_NETWORKS,
    PaperEDPConfig,
    PaperEDPReproduction,
)
from opticalloop.applications.rosa.reproduction import (
    EnvironmentDoctor,
    ExperimentManifest,
    ReproductionAnalyzer,
    ReproductionRunner,
    ReproductionValidator,
)
from opticalloop.applications.rosa.multislice import (
    ACCURACY_STATUS,
    ASWMOptimizer,
    MultiSliceAnalyzer,
    MultiSliceValidator,
    ParetoState,
    SliceChoice,
    SliceEnergyModel,
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
    "ACCURACY_STATUS",
    "ASWMOptimizer",
    "FINAL_ARTIFACTS",
    "HybridMappingSpec",
    "EnvironmentDoctor",
    "ExperimentManifest",
    "MacroVariant",
    "MultiSliceAnalyzer",
    "MultiSliceValidator",
    "PAPER_NETWORKS",
    "PaperEDPConfig",
    "PaperEDPReproduction",
    "ParetoState",
    "RosaResultValidator",
    "RosaWorkflow",
    "RosaWorkflowSpec",
    "SliceChoice",
    "SliceEnergyModel",
    "ReproductionAnalyzer",
    "ReproductionRunner",
    "ReproductionValidator",
    "ValidationCheck",
    "default_rosa_workflow",
    "parse_architecture_argument",
    "write_reference_artifacts",
]
