# Workflow Architecture

OpticalLoop separates live Timeloop execution from workflow orchestration and result validation. This keeps the simulation boundary explicit: Timeloop produces measurements, while OpticalLoop arranges experiments and checks the resulting CSV artifacts.

## Main Components

| Component | Responsibility |
| --- | --- |
| `TimeloopBackend` | Adapter for `workspace/scripts/utils.quick_run` and batch Timeloop mapper execution. Workflow code does not call Timeloop helpers directly. |
| `SimulationResult` | Normalized result object for one Timeloop-backed layer run, including cycles, latency, energy, area, and component breakdowns. |
| `RosaWorkflowSpec` | Immutable workflow configuration: networks, macro variants, architecture settings, system, output name, result directory, and job count. |
| `MacroVariant` | One macro family in a sweep, such as no-OSA `proposed_mrr` or OSA `proposed_mrr_optical_shift_add`. |
| `HybridMappingSpec` | Per-layer boolean mapping that selects a regular or hybrid macro family from `workspace/hybrid_mapping/*.yaml`. |
| `RosaWorkflow` | Orchestrates sweeps, reconstruction, aggregation, ranking, reporting, and hybrid workflows. |
| `RosaResultValidator` | Checks final artifacts against schemas, row counts, headline values, ranking formulas, and portable-path constraints. |

## Data Flow

1. Workload YAMLs and architecture definitions are read from the vendored `workspace/`.
2. `RosaWorkflow` expands networks, macro variants, and architecture settings into `TimeloopRun` objects.
3. `TimeloopBackend` forwards each run to Timeloop through `quick_run` or batch execution.
4. Timeloop stats are converted into `SimulationResult` objects.
5. Workflow writers emit raw breakdown, combined breakdown, tidy module data, and architecture metric CSVs.
6. Reconstruction and aggregation produce `results/reconstructed/*.csv`.
7. Ranking recomputes six-network OSA scores from generated metrics.
8. Artifact generation copies lightweight final CSVs into `examples/results/` and writes deterministic plots into `examples/plots/`.
9. Validation checks committed examples or regenerated artifacts against ROSA/Timeloop gold values.

## Supported ROSA Presets

The default `rosa-full` preset covers:

- AlexNet no-OSA versus OSA architecture sweep.
- Six-network OSA ranking across `alexnet`, `vgg16`, `resnet18`, `mobilenet_v3`, `gpt2_medium`, and `vision_transformer`.
- Hybrid mapping families for OSA and legacy delay-line macros when a network has a YAML mapping.

Hybrid mapping files are vendored for `alexnet`, `vgg16`, `resnet18`, `mobilenet_v3`, and `mobilenet`. The workflow skips networks that do not have a hybrid mapping file.

## Default Architecture Sweep

The default ROSA sweep uses these ten architecture settings:

| Tiles | PEs | Cols | Rows |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 9 | 113 |
| 1 | 1 | 100 | 12 |
| 1 | 64 | 4 | 4 |
| 1 | 32 | 4 | 8 |
| 1 | 16 | 4 | 16 |
| 1 | 8 | 4 | 32 |
| 1 | 32 | 8 | 4 |
| 1 | 16 | 8 | 8 |
| 1 | 8 | 8 | 16 |
| 1 | 4 | 8 | 32 |

The default no-OSA macro is `proposed_mrr`, with output postfix `_1bit_input`. The default OSA macro is `proposed_mrr_optical_shift_add`, with output postfix `_1bit_input_osa`.

## Vendored Workspace

`workspace/` contains the models and scripts required for the artifact:

- `workspace/models/workloads/`: workload YAMLs for the supported networks.
- `workspace/models/arch/`: macro, tile, chip, and system architecture templates.
- `workspace/models/components/`: component models and Accelergy plug-ins used by Timeloop.
- `workspace/scripts/`: Timeloop/CIMLoop helper scripts, including `utils.quick_run`.
- `workspace/hybrid_mapping/`: layer-level hybrid macro selection YAMLs.

This vendoring is intentional so the repo can be used as an independent artifact without relying on a parent checkout for models or scripts.
