"""Workload references for Timeloop-backed OpticalLoop runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeloopLayerRef:
    """Reference to a layer already represented in Timeloop workload YAMLs."""

    network: str
    layer_path: str

    def __post_init__(self) -> None:
        if not self.network:
            raise ValueError("network cannot be empty")
        if not self.layer_path:
            raise ValueError("layer_path cannot be empty")

    @property
    def layer_id(self) -> str:
        return self.layer_path.rsplit("/", 1)[-1].replace(".yaml", "")
