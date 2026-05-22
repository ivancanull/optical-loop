# Artifact Reproduction

This document describes how to evaluate OpticalLoop as a paper artifact. It assumes commands are run from the repository root and that the `timeloop` conda environment is available.

## Checklist

| Step | Command | Expected outcome |
| --- | --- | --- |
| Cached headline report | `conda run -n timeloop python rosa_workflow.py --stage report` | Prints the AlexNet OSA and six-network ranking headline values. |
| Cached validation | `conda run -n timeloop python rosa_workflow.py --stage validate` | Writes `examples/results/validation_report.csv` with all checks passing. |
| Unit tests | `python -m pytest -q` | Passes validation, plot, and committed-artifact tests. |
| Full rerun | `conda run -n timeloop python rosa_workflow.py --mode rerun --stage all --preset rosa-full` | Recreates the full ignored `results/` tree through Timeloop and refreshes lightweight examples. |
| Optional live smoke | `OPTICALLOOP_RUN_TIMELOOP=1 conda run -n timeloop python rosa_workflow.py --stage validate` | Adds a single live Timeloop layer check to the validation report. |

## Cached Report And Validation

The fastest artifact check reads the committed lightweight CSVs:

```bash
conda run -n timeloop python rosa_workflow.py --stage report
```

Expected headline values:

```text
AlexNet OSA best: T1,P1,C100,R12 EDP=0.0810225695
Six-network OSA best: T1, P32, C8, R4 score=0.8529673580 rank=1
```

Then run the validator:

```bash
conda run -n timeloop python rosa_workflow.py --stage validate
```

The validator writes `examples/results/validation_report.csv`. Each row contains the source CSV, check name, expected value, actual value, tolerance, pass/fail status, and optional details.

If validation fails, inspect rows where `passed` is false. Common causes are missing committed CSVs, schema drift, changed headline values, changed ranking formulas, or path strings that are not portable.

## Full Timeloop Rerun

Use rerun mode to regenerate live Timeloop results:

```bash
conda run -n timeloop python rosa_workflow.py --mode rerun --stage all --preset rosa-full
```

This runs the default ROSA sweeps, reconstructs raw Timeloop breakdown CSVs, aggregates metrics, ranks OSA architectures, runs supported hybrid mappings, regenerates lightweight artifacts, and validates the final examples. The full `results/` directory is ignored by git because it is larger and machine-generated.

To run only the Timeloop sweeps and postpone post-processing:

```bash
conda run -n timeloop python rosa_workflow.py --mode rerun --stage run --preset rosa-full
```

To run post-processing on an existing full `results/` tree:

```bash
conda run -n timeloop python rosa_workflow.py --stage aggregate
conda run -n timeloop python rosa_workflow.py --stage rank
```

## Regenerating Lightweight Artifacts

Generate the committed CSV and PNG artifact set from the best available result source:

```bash
conda run -n timeloop python rosa_workflow.py --stage artifacts
```

Source discovery checks `results/`, then a sibling `results/`, then `examples/results/`. When only committed examples are present, the command revalidates and redraws plots from those examples.

To explicitly point at a generated full result tree:

```bash
conda run -n timeloop python rosa_workflow.py --stage artifacts --artifact-source-results-dir results
```

## Timeloop-Only Simulation Boundary

Live simulation is only allowed through `TimeloopBackend`, which calls `workspace/scripts/utils.quick_run` or batch execution around Timeloop mapper invocations. The ROSA workflow, validator, and plotting code do not estimate energy, latency, cycles, or area with a separate simulator. Cache-mode commands only read existing Timeloop/CIMLoop CSVs.

## Portability Rule

Committed docs and artifact CSVs use relative paths. Do not commit machine-specific absolute paths in generated CSVs, reports, examples, or documentation.
