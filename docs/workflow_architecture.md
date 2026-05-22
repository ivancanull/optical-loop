# Workflow Architecture

OpticalLoop separates live Timeloop execution from workflow orchestration and result validation. This keeps the simulation boundary explicit: Timeloop produces measurements, while OpticalLoop arranges ROSA experiments and checks the resulting CSV artifacts.

## ROSA Architecture Context

The ROSA paper proposes a microring-resonator optical neural network (MRR-ONN) accelerator organized as chip -> tile -> optical processing element (OPE):

- At the chip level, a mesh of tiles exchanges input, weight, and output data with off-chip DRAM and a global buffer.
- At the tile level, OPEs execute matrix-vector work for CNN layers and GEMM-style Transformer layers, with local buffers for weight voltages and output partial sums.
- At the OPE level, wavelength-division multiplexing (WDM) broadcasts modulated optical inputs across MRR array rows and columns.
- The optical shift-and-add (OSA) module accumulates bit-serial optical products before OAC/ADC conversion, reducing conversion and partial-sum traffic.

The architecture-level artifact implements this modeling path through Timeloop/CiMLoop-derived templates and calibrated component data. The paper's separate PyTorch robustness and accuracy simulations are not part of the live Timeloop workflow in this repo.

## Main Components

| Component | Responsibility |
| --- | --- |
| `TimeloopBackend` | Adapter for `workspace/scripts/utils.quick_run` and batch Timeloop mapper execution. Workflow code does not call Timeloop helpers directly. |
| `SimulationResult` | Normalized result object for one Timeloop-backed layer run, including cycles, latency, energy, area, power, and component breakdowns. |
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

## Supported ROSA Workflows

The default `rosa-full` preset aligns with the paper's three architecture-level contributions:

- **OSA energy-efficiency path:** compare no-OSA `proposed_mrr` against OSA `proposed_mrr_optical_shift_add`, where OSA reduces OAC/ADC conversion and partial-sum traffic.
- **OPE dimension sweep:** evaluate ten MRR array settings across CNN and Transformer workloads to avoid over-optimizing for a single network.
- **Hybrid mapping path:** select between WS-like regular macros and IS-like hybrid macros when a network has a `workspace/hybrid_mapping/*.yaml` file.

The default six-network ranking covers `alexnet`, `vgg16`, `resnet18`, `mobilenet_v3`, `gpt2_medium`, and `vision_transformer`. Hybrid mapping files are vendored for `alexnet`, `vgg16`, `resnet18`, `mobilenet_v3`, and `mobilenet`; the workflow skips networks that do not have a hybrid mapping file.

## Default Architecture Sweep

The default ROSA sweep uses these ten architecture settings:

| Tiles | PEs | Cols | Rows | Paper context |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 9 | 113 | DEAP-CNNs high-channel style setting |
| 1 | 1 | 100 | 12 | DEAP-CNNs wide-kernel style setting |
| 1 | 64 | 4 | 4 | Compact OPE baseline with many PEs |
| 1 | 32 | 4 | 8 | Moderate row/column candidate |
| 1 | 16 | 4 | 16 | Moderate row/column candidate |
| 1 | 8 | 4 | 32 | Moderate row/column candidate |
| 1 | 32 | 8 | 4 | Validated six-network OSA ranking winner |
| 1 | 16 | 8 | 8 | Moderate row/column candidate |
| 1 | 8 | 8 | 16 | Moderate row/column candidate |
| 1 | 4 | 8 | 32 | Moderate row/column candidate |

The default no-OSA macro is `proposed_mrr`, with output postfix `_1bit_input`. The default OSA macro is `proposed_mrr_optical_shift_add`, with output postfix `_1bit_input_osa`.

## Vendored Workspace

`workspace/` contains the models and scripts required for the artifact:

- `workspace/models/workloads/`: workload YAMLs for the supported networks.
- `workspace/models/arch/`: macro, tile, chip, and system architecture templates.
- `workspace/models/components/`: component models and Accelergy plug-ins used by Timeloop.
- `workspace/scripts/`: Timeloop/CiMLoop helper scripts, including `utils.quick_run`.
- `workspace/hybrid_mapping/`: layer-level hybrid macro selection YAMLs.

This vendoring is intentional so the repo can be used as an independent artifact without relying on a parent checkout for models or scripts.
