# Reproducing the DAC26 EDP experiments

This is the reader-facing procedure for regenerating ROSA energy, latency, and
energy-delay product results from native Timeloop simulations. Accuracy,
training, DAC noise, and thermal-noise experiments are intentionally excluded.

## Clean-checkout setup

The authoritative environment is the repository `Dockerfile`. Its upstream
Timeloop/Accelergy image is pinned by registry digest, so `latest` cannot move
the build after checkout. Docker 24+ with Compose v2 is the only host
requirement.

```bash
cd optical-loop
make doctor
```

`doctor` checks the native `timeloop-mapper`, Accelergy, Python frontends,
optical macros, all 352 workload layers, and output permissions. Start a
paper run after every row passes.

## Smoke and full experiments

```bash
# Twelve jobs: one layer from each workload, with and without OSA.
make smoke

# One deterministic batch of the 7,040-job full sweep; rerun until complete.
WORKERS=8 MAX_JOBS=256 make full-batch
```

The smoke tier proves the native mapper-to-report-to-notebook path. Full network totals require the full sweep; smoke results provide
pipeline evidence rather than network-level evidence. Full runtime and storage depend strongly on
the machine and mapper search behavior. Budget multiple CPU-hours and tens of
gigabytes, begin with four workers, and increase only after observing memory
usage. `MAX_JOBS` limits only pending jobs and exits with an `incomplete`
state. Each successful job is checkpointed, so repeating the same command
resumes rather than recomputes it.

Every run is stored under `reproduction-runs/` with an ID derived from the
manifest, tier, source commit, and toolchain provenance. A run records its full
manifest, command, timestamps, versions, and job counts. Failed jobs retain
their exception in `jobs/<job-id>.json` and are retried on resume. Results from
different manifests or toolchains are never combined.

To rebuild artifacts or revalidate a selected run:

```bash
docker compose run --rm opticalloop python3 optical_loop.py reproduce analyze \
  --run-dir reproduction-runs/<run-id>
docker compose run --rm opticalloop python3 optical_loop.py reproduce validate \
  --run-dir reproduction-runs/<run-id>
```

`analyze` regenerates the executed notebook and all artifacts; `validate`
independently regenerates tables and checks without starting Jupyter:

- `layer_results.csv`: one row per successful native mapper job;
- `network_architecture_metrics.csv`: energy, latency, and EDP totals;
- `validation.csv`: machine-readable pass/warn/fail checks;
- `REPORT.md`: provenance, headline values, paper deltas, and overall status;
- `dac26_edp_reproduction.executed.ipynb`: the executed reader notebook.

## Interpretation

`PASS` means all applicable simulator and paper checks pass.
`PASS_WITH_PAPER_GAPS` means the simulation is internally consistent and agrees
with the committed raw reference data, but disclosed paper inputs are
insufficient for at least one published claim. `FAIL` means a hard simulation,
coverage, constraint, formula, winner, or reference comparison failed.

The paper defines a robust aggregate containing an unstated lambda. The
reproduction therefore reports the unambiguous six-network geometric mean and
the absolute delta from the published 26%, 64%, and 29% reductions; it never
fits lambda. The 37% optimized-ODE claim has no supplied raw ODE configuration,
and the hybrid table lacks a complete accuracy-independent experiment
definition. These remain explicit warnings rather than copied successes.
