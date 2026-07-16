# MB-OSA and Adaptive Slice-Width Mapping

MB-OSA (Multi-Bit Optical Shift-and-Add) generalizes the paper's 1-bit temporal
input stream to 1-, 2-, and 4-bit symbols. Activations and weights remain
8-bit-quantized values; only the number of activation bits carried in each
temporal symbol changes. The 8-bit analog mode is retained as a reference.

ASWM (Adaptive Slice-Width Mapping) selects 1, 2, or 4 bits independently for
each layer while holding the physical optical core shape fixed for the entire
network. It computes the exact cumulative energy/latency Pareto frontier and
chooses the point minimizing network EDP. The analog reference is excluded
from ASWM because its accuracy and robustness tradeoff is not modeled.

## Modeling boundary

The primary DAC model follows the paper's 5.2 pJ/bit value: a `b`-bit symbol
costs `b * 5.2 pJ`, and DAC area scales linearly from the 1-bit point. Reports
also include an optimistic constant-energy-per-symbol model and a conservative
Walden model proportional to `2^b - 1`.

Passive radix-shift loss is not calibrated in the source paper. Results
therefore include 0, 0.5, and 1 dB laser-compensation sensitivity per delay
stage. The primary model uses 0 dB, matching the original ideal passive OSA
assumption.

Accuracy is intentionally vacant. Artifact columns contain null `accuracy` and
`accuracy_delta` values, `accuracy_constraint=false`, and
`accuracy_status=NOT_MODELED`. No EDP result implies an accuracy claim.

## Run the experiments

Use the pinned Docker environment from the repository root:

```bash
make multislice-smoke
WORKERS=128 make multislice-full
```

The smoke tier executes 24 jobs: six representative layers across analog-8 and
MB-OSA-1/2/4. The full tier executes 14,080 native mapper jobs: 352 layers,
ten core shapes, and four execution modes. Both commands checkpoint and resume.

Resource guidance (host and mapper version dependent):

- Allocate at least 16 GB RAM and 15 GB free disk for the pinned container,
  checkpoints, logs, plots, and executed notebook.
- Smoke normally takes several minutes on a four-core host. It proves the
  native data path and scaling checks, but does not reproduce network totals.
- Full is CPU-bound and can take hours. Start with 4-16 workers on a laptop or
  workstation; larger servers can raise `WORKERS` while monitoring memory.
- Every completed job is written atomically under the immutable run ID. If a
  command is interrupted, repeat the same command and run root to resume. Do
  not copy job JSON between run IDs or edit `run.json`.

Direct CLI equivalents are:

```bash
python3 optical_loop.py multislice doctor
python3 optical_loop.py multislice smoke --workers 4
python3 optical_loop.py multislice full --workers 128
python3 optical_loop.py multislice analyze --run-dir multislice-runs/<run-id>
python3 optical_loop.py multislice validate --run-dir multislice-runs/<run-id>
```

The generated `artifacts-multislice/` directory contains raw and sensitivity
layer data, fixed-width summaries, ASWM selections and Pareto frontiers,
shared-core rankings, workload-specific best cores, primary/sensitivity
comparisons, plots, validation, a Markdown report, and an executed notebook.
Treat `validation.csv` as the machine-readable gate and `REPORT.md` as the
reader summary. A `FAIL` in any error-severity row invalidates the run.

The checked full-run reader bundle is committed in
`examples/rosa/mb_osa_reference/`. It contains provenance, validation, network
summaries, every selected layer width, sensitivity comparisons, plots, and the
executed notebook. Large raw layer checkpoints are intentionally regenerated
by `multislice full` rather than duplicated in Git.

ASWM optimization always evaluates the complete exact cumulative Pareto
frontier. For tractable plotting and CSV size, a deterministic sample of at
most 1,000 final-frontier points per workload/core/scenario is written to
`aswm_pareto_frontiers.csv`; the selected optimum is always included, and
`frontier_total_states` records the unsampled frontier size. Sampling never
participates in selection.
The exact optimizer has a documented 2,000,000-state safety cap. Exceeding it is
a hard diagnostic rather than a trigger for approximate pruning. This cap is
an analysis resource guard and does not alter or relabel native simulations in
an already completed immutable run.

## Mapping terminology

| Mapping | Input representation | Weight representation |
| --- | --- | --- |
| Analog | One 8-bit symbol | 8-bit analog |
| Digital | Eight 1-bit temporal symbols | 8-bit analog |
| Mixed | Paper hybrid WS/IS selection | Paper-dependent |
| ASWM | Layer-adaptive 1/2/4-bit temporal symbols | 8-bit analog |
