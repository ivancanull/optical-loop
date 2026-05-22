"""Validation helpers for the DEAP-CNNs application."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

import pandas as pd
import yaml

from opticalloop.applications.deap_cnns.workflow import (
    DeapArchitectureSetting,
    DeapDeviceSpec,
    architecture_summary_dataframe,
    device_parameters_dataframe,
)


DEVICE_PARAMETER_CSV = "device_parameters_deap_cnns.csv"
ARCHITECTURE_SUMMARY_CSV = "architecture_summary_deap_cnns.csv"
VALIDATION_REPORT_CSV = "validation_report.csv"


EXPECTED_DEVICE_VALUES: Mapping[str, object] = {
    "mrr_precision_bits": 7,
    "mrr_self_coupling": 0.99,
    "mrr_loss": 0.99,
    "max_wavelengths": 100,
    "max_modulators": 1024,
    "waveguide_width_nm": 500,
    "waveguide_thickness_nm": 220,
    "waveguide_bend_radius_um": 5,
    "wavelength_min_um": 1.5,
    "wavelength_max_um": 1.6,
    "propagation_radius_um": 10,
    "propagation_mrrs": 100,
    "propagation_time_ps": 21,
    "balanced_pd_throughput_gsps": 25,
    "tia_throughput_gsps": 10,
    "mrr_modulation_throughput_gsps": 128,
    "dac_throughput_gsps": 5,
    "adc_throughput_gsps": 5,
    "output_cycle_ps": 200,
    "laser_power_mw": 100.0,
    "mrr_power_mw": 19.5,
    "dac_power_mw": 26.0,
    "tia_power_mw": 17.0,
    "adc_power_mw": 76.0,
}


@dataclass(frozen=True)
class ValidationCheck:
    source: str
    check_name: str
    expected: str
    actual: str
    tolerance: float
    passed: bool
    details: str = ""


class DeapResultValidator:
    """Validate DEAP-CNNs in-repo artifacts and device constraints."""

    def __init__(
        self,
        artifact_dir: Path = Path("examples/deap_cnns"),
        *,
        repo_root: Optional[Path] = None,
        tolerance: float = 1e-12,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.repo_root = Path(repo_root or Path.cwd())
        self.tolerance = tolerance

    def validate(self) -> pd.DataFrame:
        checks: List[ValidationCheck] = []
        checks.extend(self._validate_reference_not_tracked())
        checks.extend(self._validate_device_spec())
        checks.extend(self._validate_timeloop_asset_parameters())
        checks.extend(self._validate_architecture_constraints())
        checks.extend(self._validate_artifact_files())
        checks.extend(self._validate_no_absolute_paths())
        return pd.DataFrame([asdict(check) for check in checks])

    def assert_valid(self) -> None:
        report = self.validate()
        failed = report[~report["passed"]]
        if not failed.empty:
            raise AssertionError(f"Validation failed:\n{failed.to_string(index=False)}")

    def write_report(self, path: Path) -> Path:
        report = self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False)
        return path

    def _validate_reference_not_tracked(self) -> List[ValidationCheck]:
        result = subprocess.run(
            ["git", "ls-files", "reference"],
            cwd=self.repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        tracked = [line for line in result.stdout.splitlines() if line.strip()]
        return [
            _check(
                "git",
                "reference_not_tracked",
                "none",
                "none" if not tracked else ",".join(tracked),
                0.0,
                not tracked,
            )
        ]

    def _validate_device_spec(self) -> List[ValidationCheck]:
        device_values = asdict(DeapDeviceSpec())
        checks = []
        for name, expected in EXPECTED_DEVICE_VALUES.items():
            actual = device_values[name]
            if isinstance(expected, float):
                checks.append(
                    _numeric_check(
                        "DeapDeviceSpec",
                        f"device:{name}",
                        expected,
                        float(actual),
                        self.tolerance,
                    )
                )
            else:
                checks.append(
                    _check(
                        "DeapDeviceSpec",
                        f"device:{name}",
                        str(expected),
                        str(actual),
                        0.0,
                        actual == expected,
                    )
                )
        return checks

    def _validate_timeloop_asset_parameters(self) -> List[ValidationCheck]:
        variables_path = (
            self.repo_root
            / "workspace"
            / "models"
            / "arch"
            / "1_macro"
            / "deap_cnns"
            / "variables_iso.yaml"
        )
        checks = [
            _check(
                variables_path,
                "timeloop_variables_file_exists",
                "true",
                str(variables_path.exists()).lower(),
                0.0,
                variables_path.exists(),
            )
        ]
        if not variables_path.exists():
            return checks

        variables = yaml.safe_load(variables_path.read_text())["variables"]
        expected_values: Mapping[str, object] = {
            "SCALING": "deapcnns",
            "DEAP_MRR_PRECISION_BITS": 7,
            "DEAP_ADC_POWER_MW": 76,
            "DEAP_OUTPUT_CYCLE_PS": 200,
            "ADC_ENERGY_SCALE": 0.076 / 0.52136,
        }
        for name, expected in expected_values.items():
            actual = variables[name]
            if isinstance(expected, float):
                checks.append(
                    _numeric_check(
                        variables_path,
                        f"timeloop_variable:{name}",
                        expected,
                        float(actual),
                        self.tolerance,
                    )
                )
            else:
                checks.append(
                    _check(
                        variables_path,
                        f"timeloop_variable:{name}",
                        str(expected),
                        str(actual).strip('"'),
                        0.0,
                        str(actual).strip('"') == str(expected),
                    )
                )
        return checks

    def _validate_architecture_constraints(self) -> List[ValidationCheck]:
        checks = []
        for architecture in (
            DeapArchitectureSetting("mnist-default", 1, 5, 8),
            DeapArchitectureSetting("edge-small", 1, 3, 113),
            DeapArchitectureSetting("edge-large", 1, 10, 10),
        ):
            checks.extend(_architecture_checks(architecture))

        try:
            DeapArchitectureSetting("paper-edge-large-stated", 1, 10, 12)
        except ValueError as exc:
            checks.append(
                _check(
                    "DeapArchitectureSetting",
                    "reject_paper_edge_large_stated_1200_modulators",
                    "rejected",
                    "rejected",
                    0.0,
                    True,
                    details=str(exc),
                )
            )
        else:
            checks.append(
                _check(
                    "DeapArchitectureSetting",
                    "reject_paper_edge_large_stated_1200_modulators",
                    "rejected",
                    "accepted",
                    0.0,
                    False,
                )
            )
        return checks

    def _validate_artifact_files(self) -> List[ValidationCheck]:
        checks = []
        expected_files = {
            DEVICE_PARAMETER_CSV: {"parameter", "value", "unit", "source"},
            ARCHITECTURE_SUMMARY_CSV: {
                "name",
                "n_conv_units",
                "kernel_edge",
                "input_channels",
                "n_wavelengths",
                "n_modulators",
                "architecture",
            },
        }
        for filename, required_columns in expected_files.items():
            path = self.artifact_dir / filename
            exists = path.exists()
            checks.append(
                _check(path, "file_exists", "true", str(exists).lower(), 0.0, exists)
            )
            if not exists:
                continue
            dataframe = pd.read_csv(path)
            missing = sorted(required_columns - set(dataframe.columns))
            checks.append(
                _check(
                    path,
                    "required_columns",
                    "present",
                    "present" if not missing else f"missing:{','.join(missing)}",
                    0.0,
                    not missing,
                )
            )
        return checks

    def _validate_no_absolute_paths(self) -> List[ValidationCheck]:
        checks = []
        for path in self.artifact_dir.glob("*.csv"):
            offenders = _absolute_path_values(pd.read_csv(path))
            checks.append(
                _check(
                    path,
                    "no_absolute_local_paths",
                    "none",
                    "none" if not offenders else offenders[0],
                    0.0,
                    not offenders,
                    details=f"{len(offenders)} offending values",
                )
            )
        return checks


def write_deap_artifacts(
    artifact_dir: Path = Path("examples/deap_cnns"),
    *,
    validation_report_name: str = VALIDATION_REPORT_CSV,
) -> Mapping[str, Path]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device_path = artifact_dir / DEVICE_PARAMETER_CSV
    architecture_path = artifact_dir / ARCHITECTURE_SUMMARY_CSV
    report_path = artifact_dir / validation_report_name

    device_parameters_dataframe().to_csv(device_path, index=False)
    architecture_summary_dataframe().to_csv(architecture_path, index=False)
    validator = DeapResultValidator(artifact_dir)
    validator.write_report(report_path)
    validator.assert_valid()
    return {
        DEVICE_PARAMETER_CSV: device_path,
        ARCHITECTURE_SUMMARY_CSV: architecture_path,
        validation_report_name: report_path,
    }


def _architecture_checks(
    architecture: DeapArchitectureSetting,
) -> List[ValidationCheck]:
    device = DeapDeviceSpec()
    return [
        _check(
            "DeapArchitectureSetting",
            f"architecture:{architecture.name}:wavelength_limit",
            f"<= {device.max_wavelengths}",
            str(architecture.n_wavelengths),
            0.0,
            architecture.n_wavelengths <= device.max_wavelengths,
        ),
        _check(
            "DeapArchitectureSetting",
            f"architecture:{architecture.name}:modulator_limit",
            f"<= {device.max_modulators}",
            str(architecture.n_modulators),
            0.0,
            architecture.n_modulators <= device.max_modulators,
        ),
    ]


def _numeric_check(
    source: object,
    check_name: str,
    expected: float,
    actual: float,
    tolerance: float,
) -> ValidationCheck:
    return _check(
        source,
        check_name,
        f"{expected:.12g}",
        f"{actual:.12g}",
        tolerance,
        math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
    )


def _check(
    source: object,
    check_name: str,
    expected: str,
    actual: str,
    tolerance: float,
    passed: bool,
    details: str = "",
) -> ValidationCheck:
    return ValidationCheck(
        source=_portable_source(source),
        check_name=check_name,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
        passed=bool(passed),
        details=details,
    )


def _portable_source(source: object) -> str:
    if not isinstance(source, Path):
        return str(source)
    if not source.is_absolute():
        return source.as_posix()
    try:
        return source.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return source.name


def _absolute_path_values(dataframe: pd.DataFrame) -> List[str]:
    offenders = []
    pattern = re.compile(r"(^|[_\s,])/(home|Users|tmp|var|mnt|opt)/")
    for column in dataframe.select_dtypes(include=["object"]).columns:
        for value in dataframe[column].dropna().astype(str):
            if pattern.search(value):
                offenders.append(f"{column}={value}")
    return offenders
