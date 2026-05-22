# Results And Artifacts

This document describes the reusable result artifacts produced by OpticalLoop and the committed ROSA example artifacts.

## Generic Result Outputs

Application workflows can use `workflow/results.py` to write:

| Output | Meaning |
| --- | --- |
| `results_<save_name>_<network>_breakdown<postfix>.csv` | Per-layer Timeloop metrics and component breakdowns. |
| `results_<save_name>_<network>_combined<postfix>.csv` | Per-layer metrics with grouped component names when an application supplies groups. |
| `architecture_metrics_<save_name>_<network><postfix>.csv` | Architecture-level totals across layers. |
| `reconstructed/results_<network>_breakdown<postfix>.csv` | Parsed breakdown rows with network, layer, and architecture columns. |
| `reconstructed/aggregated_metrics_<network><postfix>.csv` | Aggregated EDP, TOPS, energy, area, cycles, latency, power, and efficiency metrics. |

## Committed ROSA Artifacts

| File | Meaning |
| --- | --- |
| `examples/rosa/results/aggregated_metrics_alexnet_1bit_input.csv` | AlexNet no-OSA aggregate metrics across the ten default architectures. |
| `examples/rosa/results/aggregated_metrics_alexnet_1bit_input_osa.csv` | AlexNet OSA aggregate metrics across the ten default architectures. |
| `examples/rosa/results/aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv` | Final six-network OSA architecture ranking. |
| `examples/rosa/results/detailed_architecture_scores_by_network_1bit_input_osa_all_cached_networks.csv` | Per-network OSA scores used to build the final ranking. |
| `examples/rosa/results/validation_report.csv` | Validation report for schemas, formulas, headline values, and path portability. |
| `examples/rosa/plots/alexnet_osa_edp_comparison.png` | AlexNet no-OSA vs OSA EDP comparison. |
| `examples/rosa/plots/six_network_osa_ranking.png` | Six-network OSA aggregate score ranking. |

## Committed DEAP-CNNs Artifacts

| File | Meaning |
| --- | --- |
| `examples/deap_cnns/device_parameters_deap_cnns.csv` | Device constants extracted from the DEAP-CNNs article and used by the application. |
| `examples/deap_cnns/architecture_summary_deap_cnns.csv` | Supported DEAP-CNNs architecture presets and derived wavelength/modulator counts. |
| `examples/deap_cnns/validation_report.csv` | Validation report for device constants, architecture constraints, tracked-reference checks, and path portability. |

## ROSA Validation Formula

For each network and architecture:

```text
Combined_Score = Relative_Latency^1.0 * Relative_Energy_per_MAC^1.5
```

For each architecture across the six networks:

```text
Aggregated_Score = 0.75 * geometric_mean(Combined_Score) + 0.25 * worst_case(Combined_Score)
```

The best architecture is the row with the lowest aggregate score.

## Glossary

| Term | Meaning |
| --- | --- |
| ADC | Analog-to-digital converter after optical/electrical accumulation. |
| EDP | Energy-delay product, combining energy and latency. |
| IS | Input-stationary mapping. |
| MRR | Microring resonator. |
| OAC | Optical-to-analog conversion stage before digitization. |
| OPE | Optical processing element. |
| OSA | Optical shift-and-add. |
| WDM | Wavelength-division multiplexing. |
| WS | Weight-stationary mapping. |
