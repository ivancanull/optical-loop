"""Workload references for Timeloop-backed OpticalLoop runs."""

from dataclasses import dataclass


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


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


@dataclass(frozen=True)
class LinearLayerConfig:
    """Legacy metadata for dense layers.

    Production OpticalLoop simulations do not use this class to compute energy,
    cycles, or latency. Those values must come from Timeloop outputs.
    """

    batch_size: int
    in_features: int
    out_features: int
    input_bits: int
    weight_bits: int
    output_bits: int

    def __post_init__(self) -> None:
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("in_features", self.in_features)
        _require_positive_int("out_features", self.out_features)
        _require_positive_int("input_bits", self.input_bits)
        _require_positive_int("weight_bits", self.weight_bits)
        _require_positive_int("output_bits", self.output_bits)

    @property
    def macs(self) -> int:
        return self.batch_size * self.in_features * self.out_features

    @property
    def input_values(self) -> int:
        return self.batch_size * self.in_features

    @property
    def weight_values(self) -> int:
        return self.in_features * self.out_features

    @property
    def output_values(self) -> int:
        return self.batch_size * self.out_features
