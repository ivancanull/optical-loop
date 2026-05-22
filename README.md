# OpticalLoop

OpticalLoop is a Timeloop-backed simulator framework for optical accelerator studies. It provides a small Python core for describing optical macro configurations, running Timeloop mapper jobs, normalizing results, writing reusable CSV artifacts, and building application workflows on top of those primitives. Live simulation is Timeloop-only: OpticalLoop does not introduce a local analytic energy, latency, cycle, or area simulator.

The first application in this repository is **ROSA**, an included example for microring-resonator optical neural networks (MRR-ONNs) with optical shift-and-add (OSA) and layer-wise hybrid mapping. ROSA lives under `applications/rosa/`; it is not the boundary of the simulator itself.

## Quick Start

Run the committed ROSA application report:

```bash
conda run -n timeloop python optical_loop.py rosa --stage report
```

Validate the committed ROSA example artifacts:

```bash
conda run -n timeloop python optical_loop.py rosa --stage validate
```

Run one generic Timeloop-backed layer simulation:

```bash
conda run -n timeloop python optical_loop.py layer \
  --network alexnet \
  --layer alexnet/0 \
  --macro proposed_mrr_optical_shift_add \
  --tiles 1 --pes 1 --cols 100 --rows 12
```

Run tests:

```bash
python -m pytest -q
```

The repository directory is named `optical-loop`, but the CLI bootstraps the local Python package as `opticalloop`, so an editable install is not required for these commands.

## Core Simulator

| Path | Purpose |
| --- | --- |
| `backend.py` | Timeloop adapter boundary. It owns live `quick_run` and batch mapper execution. |
| `result.py` | Normalized `SimulationResult` objects from Timeloop stats. |
| `config/` | Architecture and workload reference dataclasses passed to Timeloop. |
| `simulator/` | Layer-level simulator facade with optional cache lookup. |
| `workflow/results.py` | Reusable CSV writing, reconstruction, and aggregation utilities. |
| `module_data.py` | Tidy per-module energy, area, and power rows from Timeloop results. |
| `workspace/` | Vendored Timeloop models, workloads, macro definitions, scripts, and hybrid mappings. |

The public core API exports `TimeloopBackend`, `TimeloopRun`, `SimulationResult`, `MRRMacroConfig`, `TimeloopLayerRef`, `LayerSimulator`, `TimeloopResultCache`, and module-data helpers from `opticalloop`.

## Applications

| Application | Location | What it provides |
| --- | --- | --- |
| ROSA | `applications/rosa/` | MRR-ONN architecture sweeps, no-OSA vs OSA comparison, six-network OSA ranking, hybrid mapping workflow, validation, and plots. |

ROSA lightweight artifacts live under `examples/rosa/results/` and `examples/rosa/plots/`. The committed ROSA checks currently validate:

| Check | Expected result |
| --- | --- |
| AlexNet OSA best EDP | `T1,P1,C100,R12`, EDP `0.0810225695` |
| Six-network OSA ranking winner | `T1, P32, C8, R4`, score `0.8529673580` |

The original ROSA study also included PyTorch behavioral experiments for DAC/thermal noise, 8-bit quantized accuracy, and CIFAR-10 robustness. Those algorithm-level experiments are not part of this repository's live simulation path, which remains Timeloop-backed architecture evaluation.

## Common Commands

ROSA cached report and validation:

```bash
conda run -n timeloop python optical_loop.py rosa --stage report
conda run -n timeloop python optical_loop.py rosa --stage validate
```

Regenerate ROSA lightweight examples from an in-repo full result tree:

```bash
conda run -n timeloop python optical_loop.py rosa --stage artifacts
```

Run the full ROSA architecture workflow with live Timeloop mapper calls:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage all --preset rosa-full
```

Run only the ROSA hybrid mapping workflow:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage hybrid --hybrid-family both
```

## Documentation

- `docs/simulator_overview.md`: simulator architecture, API boundaries, and Timeloop-only correctness model.
- `docs/cli_and_api.md`: command-line and Python API usage.
- `docs/rosa_application.md`: ROSA application workflow and validation.
- `docs/results_and_artifacts.md`: result files, validation formulas, artifact layout, and glossary.
