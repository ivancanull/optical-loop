# OpticalLoop

OpticalLoop is a self-contained, Timeloop-backed artifact repository for reproducing ROSA/CIMLoop-style optical MRR accelerator experiments. Live simulation is intentionally limited to `TimeloopBackend`, which forwards runs to `workspace/scripts/utils.quick_run` and the Timeloop mapper; the surrounding Python code only orchestrates runs, reconstructs CSVs, validates reference values, and generates plots.

## Artifact Highlights

The committed lightweight reference artifacts reproduce the current ROSA headline checks:

| Check | Expected result |
| --- | --- |
| AlexNet OSA best EDP | `T1,P1,C100,R12`, EDP `0.0810225695` |
| Six-network OSA ranking winner | `T1, P32, C8, R4`, score `0.8529673580` |

Reference CSVs live in `examples/results/` and reference plots live in `examples/plots/`. The full generated `results/` tree is intentionally ignored and can be regenerated on demand.

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

Run the full ROSA workflow with live Timeloop mapper calls:

```bash
conda run -n timeloop python rosa_workflow.py --mode rerun --stage all --preset rosa-full
```

Run only the hybrid mapping workflow:

```bash
conda run -n timeloop python rosa_workflow.py --mode rerun --stage hybrid --hybrid-family both
```

`cache` mode reads existing CSVs and committed examples. Any command that generates new simulation results must use `--mode rerun`, which routes live runs through Timeloop.

## Documentation

- `docs/artifact_reproduction.md`: paper-artifact checklist and exact reproduction commands.
- `docs/workflow_architecture.md`: workflow components, data flow, supported networks, macros, and architecture sweep.
- `docs/results_and_validation.md`: committed result files, validation formulas, schemas, and plot interpretation.

## Correctness Boundary

OpticalLoop does not introduce a local analytic energy, latency, or cycle simulator for the ROSA workflow. Timeloop is the simulation source of truth. Cache-mode commands are post-processing and validation of existing Timeloop/CIMLoop CSVs; rerun-mode commands invoke Timeloop through the backend adapter.
