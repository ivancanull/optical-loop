# Results And Validation

This document catalogs the committed OpticalLoop reference artifacts and explains how validation proves they match the ROSA/Timeloop gold values.

## Committed Result Files

| File | Meaning |
| --- | --- |
| `examples/results/aggregated_metrics_alexnet_1bit_input.csv` | AlexNet no-OSA aggregate metrics across the ten default architectures. |
| `examples/results/aggregated_metrics_alexnet_1bit_input_osa.csv` | AlexNet OSA aggregate metrics across the ten default architectures. |
| `examples/results/aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv` | Final six-network OSA architecture ranking. |
| `examples/results/detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv` | Per-network OSA scores used to build the final ranking. |
| `examples/results/validation_report.csv` | Provenance-rich validation report for schemas, formulas, headline values, and path portability. |

## Committed Plot Files

| File | Meaning |
| --- | --- |
| `examples/plots/alexnet_osa_edp_comparison.png` | Bar chart comparing AlexNet no-OSA and OSA EDP across the ten default architectures. |
| `examples/plots/six_network_osa_ranking.png` | Bar chart of final six-network OSA aggregate scores, where lower is better. |

`examples/plots/alexnet_best_osa_module_energy.png` is generated only when a tidy module data CSV named `module_data_deapcnns_alexnet_1bit_input_osa.csv` is present in the artifact results directory.

## Required CSV Shapes

The AlexNet aggregate metric CSVs must contain ten rows and include:

```text
Tiles, PEs, Cols, Rows, EDP, TOPS, Energy, Area, Cycles, Latency,
Power, Energy_per_MAC, TOPS_per_W, TOPS_per_mm2
```

The final ranking CSV must contain ten rows and include:

```text
Architecture, Aggregated_Score, Rank, Geometric_Mean, Worst_Case_Score,
Best_Network, Best_Network_Score, Worst_Network, Worst_Network_Score
```

The detailed six-network score CSV must contain sixty rows and include:

```text
Network, Architecture, Tiles, PEs, Cols, Rows, Latency, Energy_per_MAC,
EDP, Relative_Latency, Relative_Energy_per_MAC, Combined_Score
```

## Validation Targets

The validator checks these headline values with absolute and relative tolerance `1e-9`:

| Check | Expected value |
| --- | --- |
| AlexNet OSA best architecture | `T1,P1,C100,R12` |
| AlexNet OSA best EDP | `0.0810225695` |
| Six-network best architecture | `T1, P32, C8, R4` |
| Six-network best score | `0.8529673580` |

The validator also checks that committed final CSVs do not contain machine-specific absolute path strings.

## Ranking Formula

For each network and architecture, the detailed score is recomputed as:

```text
Combined_Score = Relative_Latency^1.0 * Relative_Energy_per_MAC^1.5
```

For each architecture across the six networks, the aggregate score is recomputed as:

```text
Aggregated_Score = 0.75 * geometric_mean(Combined_Score) + 0.25 * worst_case(Combined_Score)
```

The best architecture is the row with the lowest aggregate score.

## Running Validation

Run:

```bash
conda run -n timeloop python rosa_workflow.py --stage validate
```

Expected success message:

```text
Validation passed (90 checks); wrote examples/results/validation_report.csv
```

The report columns are:

```text
source_csv, check_name, expected, actual, tolerance, passed, details
```

Rows where `passed` is false identify the exact artifact, check, expected value, and actual value that diverged.

## Plot Interpretation

`alexnet_osa_edp_comparison.png` compares no-OSA and OSA EDP for the same architecture labels. It is intended as a quick visual check that the OSA sweep preserves the same architecture set and exposes the AlexNet best point.

`six_network_osa_ranking.png` displays the aggregate architecture scores after applying the six-network formula. Lower bars are better; the leftmost ranked winner should be `T1, P32, C8, R4`.
