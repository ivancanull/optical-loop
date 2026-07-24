# Modeling Optical Hardware And Dataflows

OpticalLoop is organized as a workload-aware, system-level optical computing simulator with reusable application presets layered on top. The core package describes workload layers and optical/electronic hierarchies, executes native Timeloop mappings, and preserves cycles, data movement, component actions, memory activity, conversion costs, and mapper loop text. An optional accuracy package applies MRR variation to a quantized network and evaluates an explicit mapping policy.

## Simulation Boundary

Performance simulation always flows through:

```text
workload + architecture + mapping constraints
    -> LayerSimulator -> TimeloopBackend -> native Timeloop mapper
    -> cycles, energy, latency, area, component activity, mapping text
```

The performance path derives runtime behavior from Timeloop/Accelergy scheduled actions across the optical core, converters, electronic peripherals, and memory hierarchy. Layer runs invoke the native mapper; an explicitly configured result cache can reuse previously generated Timeloop rows.

The optional accuracy boundary is independent:

```text
layer policy + MRR thermal/DAC variation
    -> AccuracyBackend -> quantized whole-network inference
    -> accuracy distribution and accuracy delta
```

A joint optimizer combines whole-network accuracy with network EDP, defined as total layer energy multiplied by total layer latency. The current accuracy runtime is limited to the documented quantized ResNet18 1-bit/hybrid semantics.

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

Applications can define mapping searches, presets, workflow stages, validation rules, plots, and committed examples while reusing the simulator core. ROSA and MB-OSA/ASWM are validation examples; DEAP-CNNs demonstrates the same generic backend with a different macro, explicit workloads, and explicit variables.

## Workspace

`workspace/` is vendored so the repository can run independently:

- `workspace/models/workloads/`: Timeloop workload YAMLs.
- `workspace/models/arch/`: macro, tile, chip, and system templates.
- `workspace/models/components/`: component models and Accelergy plug-ins.
- `workspace/scripts/`: Timeloop helper scripts, including `utils.quick_run`.
- `workspace/hybrid_mapping/`: ROSA layer-wise hybrid mapping YAMLs.

## Portability

Generated full result trees stay in ignored `results/` directories. Committed examples are small, portable, and use repository-relative paths.

## Development Rules

Project-level development rules live in `docs/development_guidelines.md`. In short: keep the design object-oriented but simple, keep raw Timeloop and external-library calls inside adapter layers, and update tests when behavior changes.
## Canonical MRR Dataflows

The research model has exactly four ROSA MRR macros. Stationarity and temporal
accumulation are independent choices rather than encoded in historical names.

| Macro | Stationarity | Optical order | Accumulator |
| --- | --- | --- | --- |
| `mrr_ws_no_osa` | Weight stationary | sliced input → 8-bit weight | Digital shift-add |
| `mrr_ws_osa` | Weight stationary | sliced input → 8-bit weight | Optical delay line |
| `mrr_is_no_osa` | Input stationary | sliced weight → 8-bit input | Digital shift-add |
| `mrr_is_osa` | Input stationary | sliced weight → 8-bit input | Optical delay line |

WS keeps weights at the weight DAC/MRR and uses the paper WS spatial constraints.
Its accumulator traverses output dimension `X`. IS keeps inputs at the input
DAC, streams sliced weights through the first MRR, and maps only output channel
`M` across photonic PEs so those PEs reuse the stationary input. Its accumulator
traverses `Y`. These properties must be visible in native mapper loop text.
All macros accept the same public variables:

```text
FRONT_MRR_SLICE_BITS = 1 | 2 | 4 | 8
FRONT_MRR_RADIX = 2^FRONT_MRR_SLICE_BITS
N_TEMPORAL_SLICES = 8 / FRONT_MRR_SLICE_BITS
N_TEMPORAL_ACCUMULATIONS = N_TEMPORAL_SLICES - 1
```

For WS, `FRONT_MRR_SLICE_BITS` controls the input DAC/MRR and weight remains
8-bit. For IS it controls the weight DAC/MRR and input remains 8-bit. MRR
instance counts and core dimensions never depend on the slice width.

At 8 bits the accumulator is disabled, giving the Analog mapping. At smaller
widths, OSA macros retain partial sums in the delay line. No-OSA macros convert
each partial result and use the digital shift-add component; its energy scale
accounts for `N_TEMPORAL_SLICES - 1` actual additions rather than charging the
initial load as an addition.

`deap_cnns` is the separate generic row/column example outside the four-entry
ROSA mapping space.
