"""Validation helpers for ROSA/Timeloop result artifacts."""

from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from opticalloop.config.architecture import MRRMacroConfig
from opticalloop.config.workload import TimeloopLayerRef
from opticalloop.simulator.layer_simulator import LayerSimulator
from opticalloop.workflow.plots import write_reference_plots


TOLERANCE = 1e-9

FINAL_ARTIFACTS: Mapping[str, Path] = {
    "aggregated_metrics_alexnet_1bit_input.csv": Path(
        "reconstructed/aggregated_metrics_alexnet_1bit_input.csv"
    ),
    "aggregated_metrics_alexnet_1bit_input_osa.csv": Path(
        "reconstructed/aggregated_metrics_alexnet_1bit_input_osa.csv"
    ),
    "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv": Path(
        "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv"
    ),
    "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv": Path(
        "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv"
    ),
}

AGGREGATED_COLUMNS = {
    "Tiles",
    "PEs",
    "Cols",
    "Rows",
    "EDP",
    "TOPS",
    "Energy",
    "Area",
    "Cycles",
    "Latency",
    "Power",
    "Energy_per_MAC",
    "TOPS_per_W",
    "TOPS_per_mm2",
}
RANKING_COLUMNS = {
    "Architecture",
    "Aggregated_Score",
    "Rank",
    "Geometric_Mean",
    "Worst_Case_Score",
    "Best_Network",
    "Best_Network_Score",
    "Worst_Network",
    "Worst_Network_Score",
}
DETAILED_COLUMNS = {
    "Network",
    "Architecture",
    "Tiles",
    "PEs",
    "Cols",
    "Rows",
    "Latency",
    "Energy_per_MAC",
    "EDP",
    "Relative_Latency",
    "Relative_Energy_per_MAC",
    "Combined_Score",
}


@dataclass(frozen=True)
class ValidationCheck:
    source_csv: str
    check_name: str
    expected: str
    actual: str
    tolerance: float
    passed: bool
    details: str = ""


class RosaResultValidator:
    """Validate final ROSA result artifacts against known Timeloop outputs."""

    def __init__(
        self,
        results_dir: Path = Path("examples/results"),
        *,
        tolerance: float = TOLERANCE,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.tolerance = tolerance

    def validate(self) -> pd.DataFrame:
        checks: List[ValidationCheck] = []
        checks.extend(self._validate_required_files())
        checks.extend(self._validate_alexnet_osa_best())
        checks.extend(self._validate_six_network_ranking())
        checks.extend(self._validate_recomputed_ranking_formula())
        checks.extend(self._validate_no_absolute_paths())
        if os.environ.get("OPTICALLOOP_RUN_TIMELOOP"):
            checks.extend(self._validate_live_timeloop_smoke())
        return pd.DataFrame([asdict(check) for check in checks])

    def write_report(self, path: Path) -> Path:
        report = self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False)
        return path

    def assert_valid(self) -> None:
        report = self.validate()
        failed = report[~report["passed"]]
        if not failed.empty:
            raise AssertionError(f"Validation failed:\n{failed.to_string(index=False)}")

    def _validate_required_files(self) -> List[ValidationCheck]:
        checks = []
        expected_rows = {
            "aggregated_metrics_alexnet_1bit_input.csv": 10,
            "aggregated_metrics_alexnet_1bit_input_osa.csv": 10,
            "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv": 10,
            "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv": 60,
        }
        expected_columns = {
            "aggregated_metrics_alexnet_1bit_input.csv": AGGREGATED_COLUMNS,
            "aggregated_metrics_alexnet_1bit_input_osa.csv": AGGREGATED_COLUMNS,
            "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv": RANKING_COLUMNS,
            "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv": DETAILED_COLUMNS,
        }
        for filename, row_count in expected_rows.items():
            path = self.results_dir / filename
            exists = path.exists()
            checks.append(
                _check(
                    path,
                    "file_exists",
                    "true",
                    str(exists).lower(),
                    0.0,
                    exists,
                )
            )
            if not exists:
                continue
            df = pd.read_csv(path)
            checks.append(
                _check(
                    path,
                    "row_count",
                    str(row_count),
                    str(len(df)),
                    0.0,
                    len(df) == row_count,
                )
            )
            missing = sorted(expected_columns[filename] - set(df.columns))
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

    def _validate_alexnet_osa_best(self) -> List[ValidationCheck]:
        path = self.results_dir / "aggregated_metrics_alexnet_1bit_input_osa.csv"
        df, checks = _read_csv_for_check(path, AGGREGATED_COLUMNS)
        if df is None:
            return checks
        best = df.sort_values("EDP", kind="mergesort").iloc[0]
        architecture = _architecture_from_row(best, compact=True)
        checks.extend(
            [
                _check(
                    path,
                    "alexnet_osa_best_architecture",
                    "T1,P1,C100,R12",
                    architecture,
                    0.0,
                    architecture == "T1,P1,C100,R12",
                ),
                _numeric_check(
                    path,
                    "alexnet_osa_best_edp",
                    0.0810225695,
                    float(best["EDP"]),
                    self.tolerance,
                ),
            ]
        )
        return checks

    def _validate_six_network_ranking(self) -> List[ValidationCheck]:
        path = self.results_dir / (
            "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv"
        )
        ranking, checks = _read_csv_for_check(path, RANKING_COLUMNS)
        if ranking is None:
            return checks
        ranking = ranking.sort_values("Rank", kind="mergesort")
        best = ranking.iloc[0]
        checks.extend(
            [
                _check(
                    path,
                    "six_network_best_architecture",
                    "T1, P32, C8, R4",
                    str(best["Architecture"]),
                    0.0,
                    str(best["Architecture"]) == "T1, P32, C8, R4",
                ),
                _numeric_check(
                    path,
                    "six_network_best_score",
                    0.8529673580,
                    float(best["Aggregated_Score"]),
                    self.tolerance,
                ),
            ]
        )
        return checks

    def _validate_recomputed_ranking_formula(self) -> List[ValidationCheck]:
        detailed_path = self.results_dir / (
            "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv"
        )
        ranking_path = self.results_dir / (
            "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv"
        )
        detailed, detailed_checks = _read_csv_for_check(detailed_path, DETAILED_COLUMNS)
        ranking, ranking_checks = _read_csv_for_check(ranking_path, RANKING_COLUMNS)
        checks = [*detailed_checks, *ranking_checks]
        if detailed is None or ranking is None:
            return checks
        ranking = ranking.set_index("Architecture")
        for row_index, row in detailed.iterrows():
            expected_combined = float(row["Relative_Latency"]) * (
                float(row["Relative_Energy_per_MAC"]) ** 1.5
            )
            checks.append(
                _numeric_check(
                    detailed_path,
                    (
                        "combined_score_formula:"
                        f"{row['Network']}:{row['Architecture']}:{row_index}"
                    ),
                    expected_combined,
                    float(row["Combined_Score"]),
                    self.tolerance,
                )
            )
        for architecture, group in detailed.groupby("Architecture", sort=False):
            scores = group["Combined_Score"].astype(float)
            geometric_mean = math.prod(scores) ** (1.0 / len(scores))
            worst_case = scores.max()
            expected = geometric_mean * 0.75 + worst_case * 0.25
            if architecture not in ranking.index:
                checks.append(
                    _check(
                        ranking_path,
                        f"ranking_formula:{architecture}",
                        "architecture present",
                        "missing",
                        0.0,
                        False,
                    )
                )
                continue
            actual = float(ranking.loc[architecture, "Aggregated_Score"])
            checks.append(
                _numeric_check(
                    ranking_path,
                    f"ranking_formula:{architecture}",
                    expected,
                    actual,
                    self.tolerance,
                )
            )
        return checks

    def _validate_no_absolute_paths(self) -> List[ValidationCheck]:
        checks = []
        for filename in FINAL_ARTIFACTS:
            path = self.results_dir / filename
            if not path.exists():
                continue
            df = pd.read_csv(path)
            offenders = _absolute_path_values(df)
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

    def _validate_live_timeloop_smoke(self) -> List[ValidationCheck]:
        path = Path("timeloop-live")
        result = LayerSimulator(
            layer=TimeloopLayerRef(network="alexnet", layer_path="alexnet/0"),
            architecture=MRRMacroConfig(
                n_tiles=1,
                n_pes=1,
                n_cols=100,
                n_rows=12,
                max_utilization=False,
            ),
        ).run()
        return [
            _numeric_check(path, "live_smoke_cycles", 55756800, result.cycles, 0.0),
            _numeric_check(
                path,
                "live_smoke_energy",
                0.12273900629943034,
                result.energy_j,
                self.tolerance,
            ),
            _numeric_check(
                path,
                "live_smoke_latency",
                0.01115136,
                result.latency_s,
                self.tolerance,
            ),
        ]


def write_reference_artifacts(
    *,
    source_results_dir: Path,
    output_results_dir: Path = Path("examples/results"),
    output_plots_dir: Path = Path("examples/plots"),
    validation_report_name: str = "validation_report.csv",
) -> Mapping[str, Path]:
    """Copy lightweight gold CSVs, validate them, and generate plots."""

    source_results_dir = Path(source_results_dir)
    output_results_dir = Path(output_results_dir)
    output_plots_dir = Path(output_plots_dir)
    output_results_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for output_name, source_relative_path in FINAL_ARTIFACTS.items():
        source = _find_source_artifact(
            source_results_dir,
            output_name=output_name,
            source_relative_path=source_relative_path,
        )
        if not source.exists():
            raise FileNotFoundError(source)
        destination = output_results_dir / output_name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        copied[output_name] = destination

    validator = RosaResultValidator(output_results_dir)
    report_path = output_results_dir / validation_report_name
    validator.write_report(report_path)
    validator.assert_valid()
    plot_paths = write_reference_plots(output_results_dir, output_plots_dir)
    return {
        **copied,
        validation_report_name: report_path,
        **{path.name: path for path in plot_paths},
    }


def _numeric_check(
    source_csv: Path,
    check_name: str,
    expected: float,
    actual: float,
    tolerance: float,
) -> ValidationCheck:
    passed = math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
    return _check(
        source_csv,
        check_name,
        f"{expected:.12g}",
        f"{actual:.12g}",
        tolerance,
        passed,
    )


def _read_csv_for_check(
    path: Path,
    required_columns: Iterable[str],
) -> Tuple[Optional[pd.DataFrame], List[ValidationCheck]]:
    path = Path(path)
    if not path.exists():
        return None, [
            _check(
                path,
                "read_csv",
                "file present",
                "missing",
                0.0,
                False,
            )
        ]
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        return None, [
            _check(
                path,
                "read_csv_required_columns",
                "present",
                f"missing:{','.join(missing)}",
                0.0,
                False,
            )
        ]
    if df.empty:
        return None, [
            _check(
                path,
                "read_csv_non_empty",
                "non-empty",
                "empty",
                0.0,
                False,
            )
        ]
    return df, []


def _find_source_artifact(
    source_results_dir: Path,
    *,
    output_name: str,
    source_relative_path: Path,
) -> Path:
    source_results_dir = Path(source_results_dir)
    candidates = (
        source_results_dir / source_relative_path,
        source_results_dir / output_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _check(
    source_csv: Path,
    check_name: str,
    expected: str,
    actual: str,
    tolerance: float,
    passed: bool,
    details: str = "",
) -> ValidationCheck:
    return ValidationCheck(
        source_csv=_portable_path(source_csv),
        check_name=check_name,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
        passed=bool(passed),
        details=details,
    )


def _architecture_from_row(row: Mapping[str, object], *, compact: bool = False) -> str:
    separator = "," if compact else ", "
    return separator.join(
        [
            f"T{int(row['Tiles'])}",
            f"P{int(row['PEs'])}",
            f"C{int(row['Cols'])}",
            f"R{int(row['Rows'])}",
        ]
    )


def _absolute_path_values(df: pd.DataFrame) -> List[str]:
    offenders = []
    pattern = re.compile(r"(^|[_\s,])/(home|Users|tmp|var|mnt|opt)/")
    for column in df.select_dtypes(include=["object"]).columns:
        for value in df[column].dropna().astype(str):
            if pattern.search(value):
                offenders.append(f"{column}={value}")
    return offenders


def _portable_path(path: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name
