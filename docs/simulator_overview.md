# Simulator Overview

OpticalLoop is organized as a general Timeloop-backed simulator framework with application presets layered on top. The core package defines how to describe a workload layer, describe an optical macro configuration, call Timeloop, normalize mapper statistics, and write reusable result artifacts.

## Simulation Boundary

Live simulation always flows through:

```text
LayerSimulator -> TimeloopBackend -> workspace/scripts/utils.quick_run -> Timeloop mapper
```

The core code does not estimate energy, latency, cycles, area, or component breakdowns with a separate analytic simulator. Cache-mode reads existing Timeloop-generated CSVs; rerun-mode calls Timeloop.

## Core Components

| Component | Responsibility |
| --- | --- |
| `TimeloopMacroConfig` | Generic macro name, system, and explicit Timeloop variable dictionary. |
| `MRRMacroConfig` | Convenience macro shape for ROSA/MRR-style variables such as tiles, PEs, rows, columns, scaling, and DAC resolution. |
| `TimeloopLayerRef` | Reference to a workload layer already represented in `workspace/models/workloads/`. |
| `TimeloopBackend` | Adapter for `quick_run` and batch mapper execution. |
| `SimulationResult` | Normalized Timeloop stats, including energy, latency, cycles, area, TOPS, component breakdowns, and mapper loop text when available. |
| `LayerSimulator` | Layer-level facade that optionally checks a `TimeloopResultCache` before calling the backend. |
| `workflow/results.py` | Generic CSV writers, reconstruction, aggregation, and architecture metrics. |
| `module_data.py` | Tidy one-row-per-module data derived from `SimulationResult` objects. |

## Applications

Applications can define presets, workflow stages, validation rules, plots, and committed examples while reusing the simulator core. The included application workflow is ROSA. DEAP-CNNs is kept as a notebook example that uses the same generic backend with the canonical `deap_cnns` macro, explicit workload names, and explicit variables.

## Workspace

`workspace/` is vendored so the repository can run independently:

- `workspace/models/workloads/`: Timeloop workload YAMLs.
- `workspace/models/arch/`: macro, tile, chip, and system templates.
- `workspace/models/components/`: component models and Accelergy plug-ins.
- `workspace/scripts/`: Timeloop helper scripts, including `utils.quick_run`.
- `workspace/hybrid_mapping/`: ROSA layer-wise hybrid mapping YAMLs.

## Portability

Generated full result trees stay in ignored `results/` directories. Committed examples are small, portable, and must not contain machine-specific absolute paths.

## Development Rules

Project-level development rules live in `docs/development_guidelines.md`. In short: keep the design object-oriented but simple, keep raw Timeloop and external-library calls inside adapter layers, and update tests when behavior changes.
