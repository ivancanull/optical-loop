import math
from pathlib import Path

import pandas as pd
import pytest

from opticalloop.applications.rosa.plots import write_reference_plots
from opticalloop.applications.rosa.validation import RosaResultValidator


NETWORKS = (
    "alexnet",
    "vgg16",
    "resnet18",
    "mobilenet_v3",
    "gpt2_medium",
    "vision_transformer",
)

ARCHITECTURES = (
    (1, 1, 9, 113),
    (1, 1, 100, 12),
    (1, 64, 4, 4),
    (1, 32, 4, 8),
    (1, 16, 4, 16),
    (1, 8, 4, 32),
    (1, 32, 8, 4),
    (1, 16, 8, 8),
    (1, 8, 8, 16),
    (1, 4, 8, 32),
)


def test_validator_accepts_fixture_and_reports_failures(tmp_path: Path) -> None:
    _write_valid_artifacts(tmp_path)

    validator = RosaResultValidator(tmp_path)
    report = validator.validate()
    assert report["passed"].all()

    osa_path = tmp_path / "aggregated_metrics_alexnet_1bit_input_osa.csv"
    osa = pd.read_csv(osa_path)
    osa.loc[osa["Cols"] == 100, "EDP"] = 0.5
    osa.to_csv(osa_path, index=False)

    failed_report = validator.validate()
    failed = failed_report[~failed_report["passed"]]
    assert "alexnet_osa_best_edp" in set(failed["check_name"])
    with pytest.raises(AssertionError):
        validator.assert_valid()


def test_reference_plots_are_non_empty_pngs(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _write_valid_artifacts(results_dir)

    paths = write_reference_plots(results_dir, plots_dir)

    assert {path.name for path in paths} == {
        "alexnet_osa_edp_comparison.png",
        "six_network_osa_ranking.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1024


def test_committed_reference_artifacts_validate_and_are_portable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "examples" / "rosa" / "results"

    validator = RosaResultValidator(results_dir)
    report = validator.validate()

    assert report["passed"].all(), report[~report["passed"]].to_string(index=False)
    for csv_path in results_dir.glob("*.csv"):
        csv_text = csv_path.read_text()
        assert f"/{'home'}/" not in csv_text
        assert f"/{'Users'}/" not in csv_text


def _write_valid_artifacts(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _aggregated_metrics(osa=False).to_csv(
        results_dir / "aggregated_metrics_alexnet_1bit_input.csv",
        index=False,
    )
    _aggregated_metrics(osa=True).to_csv(
        results_dir / "aggregated_metrics_alexnet_1bit_input_osa.csv",
        index=False,
    )
    detailed = _detailed_scores()
    detailed.to_csv(
        results_dir
        / "detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv",
        index=False,
    )
    _ranking_scores(detailed).to_csv(
        results_dir
        / "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv",
        index=False,
    )


def _aggregated_metrics(*, osa: bool) -> pd.DataFrame:
    rows = []
    for index, (tiles, pes, cols, rows_count) in enumerate(ARCHITECTURES):
        edp = 0.2 + index * 0.05
        if osa and (tiles, pes, cols, rows_count) == (1, 1, 100, 12):
            edp = 0.0810225695
        energy = 1.0 + index
        latency = edp / energy
        area = 0.01 + index * 0.001
        cycles = 1000 + index
        power = energy / latency
        tops = 10.0 + index
        rows.append(
            {
                "Tiles": tiles,
                "PEs": pes,
                "Cols": cols,
                "Rows": rows_count,
                "EDP": edp,
                "TOPS": tops,
                "Energy": energy,
                "Area": area,
                "Cycles": cycles,
                "Latency": latency,
                "Power": power,
                "Energy_per_MAC": energy / 1000.0,
                "TOPS_per_W": tops / power,
                "TOPS_per_mm2": tops / area,
            }
        )
    return pd.DataFrame(rows)


def _detailed_scores() -> pd.DataFrame:
    rows = []
    for arch_index, architecture in enumerate(ARCHITECTURES):
        architecture_name = _architecture_name(architecture)
        for network_index, network in enumerate(NETWORKS):
            if architecture == (1, 32, 8, 4):
                score = 0.8529673580
            else:
                score = 1.0 + arch_index * 0.2 + network_index * 0.01
            rows.append(
                {
                    "Network": network,
                    "Architecture": architecture_name,
                    "Tiles": architecture[0],
                    "PEs": architecture[1],
                    "Cols": architecture[2],
                    "Rows": architecture[3],
                    "Latency": score,
                    "Energy_per_MAC": 1.0,
                    "EDP": score,
                    "Relative_Latency": score,
                    "Relative_Energy_per_MAC": 1.0,
                    "Combined_Score": score,
                }
            )
    return pd.DataFrame(rows)


def _ranking_scores(detailed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture, group in detailed.groupby("Architecture", sort=False):
        scores_by_network = group.set_index("Network")["Combined_Score"].reindex(NETWORKS)
        scores = scores_by_network.astype(float)
        geometric_mean = math.prod(scores) ** (1.0 / len(scores))
        worst_case_score = scores.max()
        best_network = scores.idxmin()
        worst_network = scores.idxmax()
        rows.append(
            {
                "Architecture": architecture,
                "Aggregated_Score": geometric_mean * 0.75 + worst_case_score * 0.25,
                "Geometric_Mean": geometric_mean,
                "Worst_Case_Score": worst_case_score,
                "Best_Network": best_network,
                "Best_Network_Score": scores[best_network],
                "Worst_Network": worst_network,
                "Worst_Network_Score": scores[worst_network],
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        "Aggregated_Score", kind="mergesort"
    )
    ranking.insert(2, "Rank", range(1, len(ranking) + 1))
    return ranking


def _architecture_name(architecture) -> str:
    tiles, pes, cols, rows = architecture
    return f"T{tiles}, P{pes}, C{cols}, R{rows}"
