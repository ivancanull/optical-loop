"""Join whole-network accuracy with Timeloop-derived design rows."""

import pandas as pd

from opticalloop.accuracy.result import AccuracyResult


def attach_accuracy(
    frame: pd.DataFrame,
    result: AccuracyResult,
    *,
    maximum_accuracy_drop: float | None = None,
) -> pd.DataFrame:
    """Return a copy with one whole-network accuracy result attached.

    Accuracy is intentionally repeated across the policy's layer/design rows:
    ONNSim measures the complete network, so it must not be interpreted as an
    independently additive per-layer metric.
    """

    output = frame.copy()
    for name, value in result.to_row().items():
        if name not in {"network", "scenario"}:
            output[name] = value
    if maximum_accuracy_drop is None:
        output["accuracy_constraint"] = False
    else:
        if maximum_accuracy_drop < 0:
            raise ValueError("maximum_accuracy_drop must be non-negative")
        delta = result.accuracy_delta
        output["accuracy_constraint"] = delta is not None and delta >= -maximum_accuracy_drop
    output["accuracy_status"] = "MODELED_WHOLE_NETWORK"
    return output
