# OpticalLoop: ROSA Artifact

OpticalLoop is the self-contained artifact repository for **ROSA: Robust and Energy-Efficient Microring-Based Optical Neural Networks via Optical Shift-and-Add and Layer-Wise Hybrid Mapping**. ROSA studies microring-resonator optical neural networks (MRR-ONNs) that combine digital-analog MACs, an optical shift-and-add (OSA) module, and layer-wise hybrid mapping to improve energy efficiency and robustness. This repository implements the architecture-level, Timeloop/CiMLoop-derived workflow for reproducing the ROSA MRR array sweeps, OSA comparisons, ranking results, hybrid mapping runs, and lightweight validation artifacts.

Live architecture simulation is intentionally limited to `TimeloopBackend`, which forwards runs to `workspace/scripts/utils.quick_run` and the Timeloop mapper. The surrounding Python code only orchestrates experiments, reconstructs CSVs, aggregates metrics, validates reference values, and generates plots.

## What This Artifact Reproduces

| Paper-facing result area | Artifact support |
| --- | --- |
| Optimized MRR OPE sizing across CNN and Transformer workloads | Default ten-architecture sweep and six-network ranking over `alexnet`, `vgg16`, `resnet18`, `mobilenet_v3`, `gpt2_medium`, and `vision_transformer`. |
| No-OSA vs OSA architecture comparison | AlexNet aggregate CSVs and `examples/plots/alexnet_osa_edp_comparison.png`. |
| OSA architecture selection across workloads | `examples/results/aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv` and `examples/plots/six_network_osa_ranking.png`. |
| Layer-wise hybrid mapping workflow | Rerun-mode support for OSA and legacy delay-line hybrid macro families using `workspace/hybrid_mapping/*.yaml`. |
| Paper-artifact correctness checks | `examples/results/validation_report.csv`, schema checks, formula checks, headline value checks, and portable-path checks. |

The committed lightweight reference artifacts reproduce these headline checks:

| Check | Expected result |
| --- | --- |
| AlexNet OSA best EDP | `T1,P1,C100,R12`, EDP `0.0810225695` |
| Six-network OSA ranking winner | `T1, P32, C8, R4`, score `0.8529673580` |

Reference CSVs live in `examples/results/` and reference plots live in `examples/plots/`. The full generated `results/` tree is intentionally ignored and can be regenerated on demand.

## What Is Not Simulated Here

The ROSA paper also includes behavioral PyTorch experiments for DAC/thermal noise, 8-bit quantized accuracy, and CIFAR-10 hybrid-mapping robustness. Those algorithm-level experiments are part of the paper narrative, but this repository's live simulation boundary is architecture-level Timeloop execution. The docs report the paper's robustness and accuracy claims for context, while the executable validation in this repo checks the Timeloop-backed architecture metrics and committed CSV artifacts.

## Quick Start

Run the cached paper-artifact report from the repository root:

```bash
conda run -n timeloop python rosa_workflow.py --stage report
```

Validate the committed reference artifacts:

```bash
conda run -n timeloop python rosa_workflow.py --stage validate
```

Run the Python tests:

```bash
python -m pytest -q
```

The repository directory is named `optical-loop`, but the CLI bootstraps the local Python package as `opticalloop`, so an editable install is not required for these commands.

## Repository Map

| Path | Purpose |
| --- | --- |
| `backend.py` | Adapter boundary for Timeloop calls. It owns all live `quick_run` and batch mapper execution. |
| `workflow/rosa.py` | ROSA workflow orchestration: sweeps, reconstruction, aggregation, ranking, and hybrid mappings. |
| `workflow/validation.py` | Artifact validation against ROSA/Timeloop gold values and formulas. |
| `workflow/plots.py` | Deterministic matplotlib plots derived from final CSV artifacts. |
| `workspace/` | Vendored Timeloop models, workloads, scripts, macro definitions, and hybrid mappings needed by this repo. |
| `examples/results/` | Committed lightweight CSV artifacts used for cached reporting and validation. |
| `examples/plots/` | Committed PNG plots generated from the final CSV artifacts. |
| `tests/` | Unit and artifact checks for validation, plotting, and path portability. |

## Main Workflow Commands

Cached report and validation:

```bash
conda run -n timeloop python rosa_workflow.py --stage report
conda run -n timeloop python rosa_workflow.py --stage validate
```

Regenerate lightweight examples from an existing full result tree:

```bash
conda run -n timeloop python rosa_workflow.py --stage artifacts
```

Run the full ROSA architecture workflow with live Timeloop mapper calls:

```bash
conda run -n timeloop python rosa_workflow.py --mode rerun --stage all --preset rosa-full
```

Run only the hybrid mapping workflow:

```bash
conda run -n timeloop python rosa_workflow.py --mode rerun --stage hybrid --hybrid-family both
```

`cache` mode reads existing CSVs and committed examples. Any command that generates new simulation results must use `--mode rerun`, which routes live runs through Timeloop.

## Documentation

- `docs/artifact_reproduction.md`: paper-artifact checklist, claim coverage, and exact reproduction commands.
- `docs/workflow_architecture.md`: ROSA architecture context, workflow components, data flow, supported networks, macros, and architecture sweep.
- `docs/results_and_validation.md`: paper result context, committed result files, validation formulas, schemas, plots, and glossary.

## Correctness Boundary

OpticalLoop does not introduce a local analytic energy, latency, or cycle simulator for the ROSA workflow. Timeloop is the simulation source of truth. Cache-mode commands are post-processing and validation of existing Timeloop/CiMLoop CSVs; rerun-mode commands invoke Timeloop through the backend adapter.
