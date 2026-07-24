"""Canonical mapping between Timeloop workloads and framework module names."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import yaml


SUPPORTED_STATIONARITY = frozenset({"WS", "IS"})
SUPPORTED_SLICE_BITS = frozenset({1, 2, 4, 8})


@dataclass(frozen=True)
class LayerBinding:
    """One physical layer expressed in both simulator namespaces."""

    layer_id: str
    workload: str
    torch_module: str

    def __post_init__(self) -> None:
        if not self.layer_id or not self.workload or not self.torch_module:
            raise ValueError("layer id, workload, and torch module cannot be empty")


@dataclass(frozen=True)
class LayerManifest:
    network: str
    dataset: str
    layers: Tuple[LayerBinding, ...]

    def __post_init__(self) -> None:
        if not self.network or not self.dataset or not self.layers:
            raise ValueError("manifest network, dataset, and layers are required")
        ids = [layer.layer_id for layer in self.layers]
        modules = [layer.torch_module for layer in self.layers]
        if len(ids) != len(set(ids)) or len(modules) != len(set(modules)):
            raise ValueError("layer ids and torch module names must be unique")

    @classmethod
    def load(cls, path: Path) -> "LayerManifest":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        layers = tuple(
            LayerBinding(
                layer_id=str(item["id"]),
                workload=str(item["workload"]),
                torch_module=str(item["torch_module"]),
            )
            for item in raw.get("layers", ())
        )
        return cls(
            network=str(raw.get("network", "")),
            dataset=str(raw.get("dataset", "")),
            layers=layers,
        )

    def module_for(self, layer_id: str) -> str:
        matches = [layer.torch_module for layer in self.layers if layer.layer_id == str(layer_id)]
        if len(matches) != 1:
            raise KeyError(f"Unknown canonical layer id: {layer_id}")
        return matches[0]


@dataclass(frozen=True)
class LayerPolicy:
    """Stationarity and temporal symbol width selected per canonical layer."""

    manifest: LayerManifest
    stationarity: Mapping[str, str]
    slice_bits: Mapping[str, int]
    name: str = "hybrid"

    def __post_init__(self) -> None:
        expected = {layer.layer_id for layer in self.manifest.layers}
        if set(self.stationarity) != expected or set(self.slice_bits) != expected:
            raise ValueError("policy must specify stationarity and slice bits for every layer")
        invalid_stationarity = set(self.stationarity.values()) - SUPPORTED_STATIONARITY
        invalid_bits = set(self.slice_bits.values()) - SUPPORTED_SLICE_BITS
        if invalid_stationarity or invalid_bits:
            raise ValueError(
                f"unsupported policy values: stationarity={sorted(invalid_stationarity)}, "
                f"slice_bits={sorted(invalid_bits)}"
            )

    def onnsim_mapping(self) -> dict[str, bool]:
        # ONNSim uses True for its input/inverse path and False for weight/normal.
        return {
            layer.torch_module: self.stationarity[layer.layer_id] == "IS"
            for layer in self.manifest.layers
        }

    def onnsim_slice_bits(self) -> dict[str, int]:
        """Express canonical slice widths in ONNSim module-name space."""
        return {
            layer.torch_module: self.slice_bits[layer.layer_id]
            for layer in self.manifest.layers
        }

    @classmethod
    def load(cls, path: Path, manifest: LayerManifest) -> "LayerPolicy":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        layer_specs = raw.get("layers", {})
        return cls(
            manifest=manifest,
            stationarity={
                str(layer_id): str(spec["stationarity"]).upper()
                for layer_id, spec in layer_specs.items()
            },
            slice_bits={
                str(layer_id): int(spec["slice_bits"])
                for layer_id, spec in layer_specs.items()
            },
            name=str(raw.get("name", "hybrid")),
        )
