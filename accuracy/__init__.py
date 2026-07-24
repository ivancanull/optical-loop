"""Optional model-accuracy support for OpticalLoop."""

from opticalloop.accuracy.backend import AccuracyBackend
from opticalloop.accuracy.config import AccuracyExperimentConfig, MRRVariationConfig
from opticalloop.accuracy.layer_manifest import LayerBinding, LayerManifest, LayerPolicy
from opticalloop.accuracy.result import AccuracyResult
from opticalloop.accuracy.joint import attach_accuracy

__all__ = [
    "AccuracyBackend",
    "AccuracyExperimentConfig",
    "AccuracyResult",
    "LayerBinding",
    "LayerManifest",
    "LayerPolicy",
    "MRRVariationConfig",
    "attach_accuracy",
]
