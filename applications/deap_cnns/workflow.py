"""Timeloop-backed DEAP-CNNs application workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from opticalloop.backend import TimeloopBackend, TimeloopRun
from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.module_data import write_module_data_csv as write_module_rows_csv
from opticalloop.result import SimulationResult
from opticalloop.workflow.results import (
    ArtifactPaths,
    aggregate_metrics_csv,
    artifact_paths,
    reconstruct_breakdown_csv,
    write_architecture_metrics_csv as write_architecture_metrics_rows_csv,
    write_results_csv,
)


DEAP_NETWORK = "deap_mnist"
DEAP_SAVE_NAME = "deap_cnns"
DEAP_MACRO = "deap_cnns"

COMPONENT_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "MRR": ("weight_mrr", "input_mrr", "mrr"),
    "DAC": ("weight_dac", "input_dac"),
    "OAC": ("laser", "photodiode_output_readout", "TIA"),
    "ADC": ("adc",),
    "Memory": ("glb", "output_buffer"),
}


@dataclass(frozen=True)
class DeapDeviceSpec:
    """Device constants extracted from the DEAP-CNNs article."""

    mrr_precision_bits: int = 7
    mrr_self_coupling: float = 0.99
    mrr_loss: float = 0.99
    max_wavelengths: int = 100
    max_modulators: int = 1024
    waveguide_width_nm: int = 500
    waveguide_thickness_nm: int = 220
    waveguide_bend_radius_um: int = 5
    wavelength_min_um: float = 1.5
    wavelength_max_um: float = 1.6
    propagation_radius_um: int = 10
    propagation_mrrs: int = 100
    propagation_time_ps: int = 21
    balanced_pd_throughput_gsps: int = 25
    tia_throughput_gsps: int = 10
    mrr_modulation_throughput_gsps: int = 128
    dac_throughput_gsps: int = 5
    adc_throughput_gsps: int = 5
    output_cycle_ps: int = 200
    laser_power_mw: float = 100.0
    mrr_power_mw: float = 19.5
    dac_power_mw: float = 26.0
    tia_power_mw: float = 17.0
    adc_power_mw: float = 76.0

    def parameter_rows(self) -> List[Dict[str, object]]:
        rows = []
        units = {
            "mrr_precision_bits": "bits",
            "waveguide_width_nm": "nm",
            "waveguide_thickness_nm": "nm",
            "waveguide_bend_radius_um": "um",
            "wavelength_min_um": "um",
            "wavelength_max_um": "um",
            "propagation_radius_um": "um",
            "propagation_time_ps": "ps",
            "balanced_pd_throughput_gsps": "GS/s",
            "tia_throughput_gsps": "GS/s",
            "mrr_modulation_throughput_gsps": "GS/s",
            "dac_throughput_gsps": "GS/s",
            "adc_throughput_gsps": "GS/s",
            "output_cycle_ps": "ps",
            "laser_power_mw": "mW",
            "mrr_power_mw": "mW",
            "dac_power_mw": "mW",
            "tia_power_mw": "mW",
            "adc_power_mw": "mW",
        }
        for name, value in asdict(self).items():
            rows.append(
                {
                    "parameter": name,
                    "value": value,
                    "unit": units.get(name, ""),
                    "source": "DEAP-CNNs article",
                }
            )
        return rows


@dataclass(frozen=True)
class DeapArchitectureSetting:
    """One DEAP-CNNs convolutional-unit architecture setting."""

    name: str
    n_conv_units: int
    kernel_edge: int
    input_channels: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        for field_name in ("n_conv_units", "kernel_edge", "input_channels"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        device = DeapDeviceSpec()
        if self.n_wavelengths > device.max_wavelengths:
            raise ValueError(
                f"{self.name} uses {self.n_wavelengths} wavelengths; "
                f"DEAP-CNNs limit is {device.max_wavelengths}"
            )
        if self.n_modulators > device.max_modulators:
            raise ValueError(
                f"{self.name} uses {self.n_modulators} modulators; "
                f"DEAP-CNNs limit is {device.max_modulators}"
            )

    @property
    def n_wavelengths(self) -> int:
        return self.kernel_edge * self.kernel_edge

    @property
    def n_modulators(self) -> int:
        return self.n_wavelengths * self.input_channels

    @property
    def output_postfix(self) -> str:
        return "_" + self.name.replace("-", "_")

    @property
    def filename_key(self) -> str:
        return (
            f"1tiles_{self.n_conv_units}pes_"
            f"{self.n_wavelengths}cols_{self.input_channels}rows"
        )

    @property
    def architecture_key(self) -> str:
        return (
            f"T1, P{self.n_conv_units}, "
            f"C{self.n_wavelengths}, R{self.input_channels}"
        )

    def to_macro_config(
        self,
        *,
        macro: str,
        system: str,
        device: DeapDeviceSpec,
    ) -> MRRMacroConfig:
        return MRRMacroConfig(
            n_tiles=1,
            n_pes=self.n_conv_units,
            n_cols=self.n_wavelengths,
            n_rows=self.input_channels,
            macro=macro,
            system=system,
            voltage_dac_resolution=device.mrr_precision_bits,
            scaling='"deapcnns"',
            max_utilization=False,
        )

    def summary_row(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_conv_units": self.n_conv_units,
            "kernel_edge": self.kernel_edge,
            "input_channels": self.input_channels,
            "n_wavelengths": self.n_wavelengths,
            "n_modulators": self.n_modulators,
            "architecture": self.architecture_key,
        }


DEAP_ARCHITECTURES: Mapping[str, DeapArchitectureSetting] = {
    "mnist-default": DeapArchitectureSetting(
        name="mnist-default",
        n_conv_units=1,
        kernel_edge=5,
        input_channels=8,
    ),
    "edge-small": DeapArchitectureSetting(
        name="edge-small",
        n_conv_units=1,
        kernel_edge=3,
        input_channels=113,
    ),
    "edge-large": DeapArchitectureSetting(
        name="edge-large",
        n_conv_units=1,
        kernel_edge=10,
        input_channels=10,
    ),
}


@dataclass(frozen=True)
class DeapWorkflowSpec:
    """Configuration for the DEAP-CNNs application workflow."""

    architecture: DeapArchitectureSetting = DEAP_ARCHITECTURES["mnist-default"]
    device: DeapDeviceSpec = DeapDeviceSpec()
    network: str = DEAP_NETWORK
    macro: str = DEAP_MACRO
    system: str = "fetch_all_lpddr4"
    save_name: str = DEAP_SAVE_NAME
    results_dir: Path = Path("results")
    n_jobs: int = 1
    repo_root: Optional[Path] = None


class DeapWorkflow:
    """Run and analyze Timeloop-backed DEAP-CNNs application artifacts."""

    def __init__(
        self,
        spec: DeapWorkflowSpec,
        backend: Optional[TimeloopBackend] = None,
    ) -> None:
        self.spec = spec
        self.backend = backend or TimeloopBackend()
        self.repo_root = (spec.repo_root or _default_repo_root()).resolve()
        self.results_dir = _resolve_path(self.repo_root, spec.results_dir)
        self.workspace_dir = self.repo_root / "workspace"
        self.models_dir = self.workspace_dir / "models"

    def run_sweeps(self) -> ArtifactPaths:
        layers = self.resolve_layers()
        results = self._run_layers(layers)
        paths = self.paths()
        self.write_breakdown_csv(results, paths.breakdown_csv)
        self.write_combined_csv(results, paths.combined_csv)
        self.write_module_data_csv(results, paths.module_data_csv)
        self.write_architecture_metrics_csv(results, paths.metrics_csv)
        return paths

    def reconstruct_and_aggregate(self) -> ArtifactPaths:
        paths = self.paths()
        reconstruct_breakdown_csv(paths.breakdown_csv, paths.reconstructed_breakdown_csv)
        aggregate_metrics_csv(
            paths.reconstructed_breakdown_csv,
            paths.aggregated_metrics_csv,
        )
        return paths

    def paths(self) -> ArtifactPaths:
        return artifact_paths(
            save_name=self.spec.save_name,
            network=self.spec.network,
            output_postfix=self.spec.architecture.output_postfix,
            results_dir=self.results_dir,
        )

    def report(self) -> Dict[str, object]:
        return {
            "application": "DEAP-CNNs",
            "macro": self.spec.macro,
            "network": self.spec.network,
            **self.spec.architecture.summary_row(),
            "device_precision_bits": self.spec.device.mrr_precision_bits,
            "output_cycle_ps": self.spec.device.output_cycle_ps,
        }

    def resolve_layers(self) -> Tuple[str, ...]:
        workload_dir = self.models_dir / "workloads" / self.spec.network
        if not workload_dir.exists():
            raise FileNotFoundError(workload_dir)
        return tuple(
            f"{self.spec.network}/{path.stem}"
            for path in sorted(workload_dir.glob("*.yaml"))
        )

    def write_breakdown_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_results_csv(results, path, group_components=False)

    def write_combined_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_results_csv(
            results,
            path,
            group_components=True,
            component_groups=COMPONENT_GROUPS,
        )

    def write_module_data_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_module_rows_csv(
            results,
            path,
            component_groups=COMPONENT_GROUPS,
        )

    def write_architecture_metrics_csv(
        self, results: Sequence[SimulationResult], path: Path
    ) -> Path:
        return write_architecture_metrics_rows_csv(results, path)

    def _run_layers(self, layers: Sequence[str]) -> List[SimulationResult]:
        macro_config = self.spec.architecture.to_macro_config(
            macro=self.spec.macro,
            system=self.spec.system,
            device=self.spec.device,
        )
        runs = [
            TimeloopRun(
                layer=TimeloopLayerRef(
                    network=self.spec.network,
                    layer_path=layer,
                ),
                architecture=macro_config,
                metadata={
                    "row_index": _row_index(
                        self.spec.macro,
                        self.spec.network,
                        layer,
                        self.spec.architecture,
                    ),
                    "layer_id": _layer_id(layer),
                    "variant_label": self.spec.architecture.name,
                },
            )
            for layer in layers
        ]
        return self.backend.run_batch(runs, n_jobs=self.spec.n_jobs)


def default_deap_workflow(
    *,
    architecture_name: str = "mnist-default",
    results_dir: Path = Path("results"),
    n_jobs: int = 1,
    backend: Optional[TimeloopBackend] = None,
) -> DeapWorkflow:
    return DeapWorkflow(
        DeapWorkflowSpec(
            architecture=deap_architecture_by_name(architecture_name),
            results_dir=results_dir,
            n_jobs=n_jobs,
        ),
        backend=backend,
    )


def deap_architecture_by_name(name: str) -> DeapArchitectureSetting:
    try:
        return DEAP_ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown DEAP-CNNs architecture: {name!r}") from exc


def device_parameters_dataframe(device: DeapDeviceSpec = DeapDeviceSpec()) -> pd.DataFrame:
    return pd.DataFrame(device.parameter_rows())


def architecture_summary_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [architecture.summary_row() for architecture in DEAP_ARCHITECTURES.values()]
    )


def _row_index(
    macro: str,
    network: str,
    layer: str,
    architecture: DeapArchitectureSetting,
) -> str:
    layer_file = f"workspace/models/workloads/{layer}.yaml"
    return f"{macro}_{network}_{layer_file}_{architecture.filename_key}"


def _layer_id(layer: str) -> str:
    return layer.rsplit("/", 1)[-1].replace(".yaml", "")


def _resolve_path(repo_root: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
