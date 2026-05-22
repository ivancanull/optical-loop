"""Deterministic plots for final ROSA result artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def write_reference_plots(results_dir: Path, plots_dir: Path) -> List[Path]:
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_alexnet_osa_edp_comparison(
            no_osa_csv=results_dir / "aggregated_metrics_alexnet_1bit_input.csv",
            osa_csv=results_dir / "aggregated_metrics_alexnet_1bit_input_osa.csv",
            output_png=plots_dir / "alexnet_osa_edp_comparison.png",
        ),
        plot_six_network_osa_ranking(
            ranking_csv=results_dir
            / "aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv",
            output_png=plots_dir / "six_network_osa_ranking.png",
        ),
    ]

    module_csv = results_dir / "module_data_deapcnns_alexnet_1bit_input_osa.csv"
    if module_csv.exists():
        paths.append(
            plot_alexnet_best_osa_module_energy(
                module_csv=module_csv,
                output_png=plots_dir / "alexnet_best_osa_module_energy.png",
            )
        )
    return paths


def plot_alexnet_osa_edp_comparison(
    *,
    no_osa_csv: Path,
    osa_csv: Path,
    output_png: Path,
) -> Path:
    no_osa = _read_architecture_metrics(no_osa_csv, "No OSA")
    osa = _read_architecture_metrics(osa_csv, "OSA")
    merged = no_osa.merge(osa, on="Architecture", suffixes=("_no_osa", "_osa"))

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    x = range(len(merged))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        merged["EDP_no_osa"],
        width=width,
        label="No OSA",
        color="#607D8B",
    )
    ax.bar(
        [i + width / 2 for i in x],
        merged["EDP_osa"],
        width=width,
        label="OSA",
        color="#00897B",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["Architecture"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("EDP")
    ax.set_title("AlexNet EDP Across MRR Architectures")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def plot_six_network_osa_ranking(*, ranking_csv: Path, output_png: Path) -> Path:
    ranking = pd.read_csv(ranking_csv).sort_values("Rank", kind="mergesort")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = range(len(ranking))
    ax.bar(
        list(x),
        ranking["Aggregated_Score"],
        color="#5E81AC",
    )
    ax.set_ylabel("Aggregated Score (lower is better)")
    ax.set_title("Six-Network OSA Architecture Ranking")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ranking["Architecture"], rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def plot_alexnet_best_osa_module_energy(
    *,
    module_csv: Path,
    output_png: Path,
) -> Path:
    data = pd.read_csv(module_csv)
    best_arch = "T1, P1, C100, R12"
    subset = data[data["architecture"] == best_arch]
    if subset.empty:
        subset = data
    grouped = (
        subset.groupby("module_group", dropna=False)["energy_j"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    grouped["module_group"] = grouped["module_group"].fillna("").replace("", "Other")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = range(len(grouped))
    ax.bar(list(x), grouped["energy_j"], color="#B48EAD")
    ax.set_ylabel("Energy (J)")
    ax.set_title("AlexNet Best OSA Module Energy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["module_group"], rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def _read_architecture_metrics(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["Architecture"] = df.apply(_architecture_from_row, axis=1)
    df["Variant"] = label
    return df


def _architecture_from_row(row) -> str:
    return f"T{int(row['Tiles'])}, P{int(row['PEs'])}, C{int(row['Cols'])}, R{int(row['Rows'])}"
