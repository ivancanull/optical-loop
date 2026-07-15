# CLI And API Usage

OpticalLoop can be used from the repository root without installation because `optical_loop.py` bootstraps the local `opticalloop` package.

## Paper reproduction CLI

Paper readers should use the pinned Docker workflow documented in
`docs/dac26_reproduction.md`. The underlying commands are:

```bash
python3 optical_loop.py reproduce doctor
python3 optical_loop.py reproduce smoke --workers 4
python3 optical_loop.py reproduce full --workers 4
python3 optical_loop.py reproduce analyze --run-dir reproduction-runs/<run-id>
python3 optical_loop.py reproduce validate --run-dir reproduction-runs/<run-id>
```

`smoke` and `full` run native simulations and then analyze, validate, and
execute the notebook. Use `--skip-notebook` only for debugging. Successful jobs
resume by default. `--no-resume` is a safety assertion that rejects an existing
provenance-keyed run; use it only with an unused `--run-root`.

## Generic Layer Command

Run one Timeloop-backed layer:

```bash
conda run -n timeloop python optical_loop.py layer \
  --workload alexnet/0 \
  --arch proposed_mrr_optical_shift_add \
  --tiles 1 --pes 1 --cols 100 --rows 12
```

Run an arbitrary macro with explicit Timeloop variables:

```bash
conda run -n timeloop python optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench0 \
  --var N_COLUMNS=100 \
  --var N_ROWS=12 \
  --var N_Conv=1 \
  --show-mapping
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--arch` | Timeloop macro name under `workspace/models/arch/1_macro/`. |
| `--workload` | Workload layer path under `workspace/models/workloads/`, such as `alexnet/0`. |
| `--var KEY=VALUE` | Generic Timeloop variable override. Repeat as needed. |
| `--system` | Timeloop system template name, default `fetch_all_lpddr4`. |
| `--tiles`, `--pes`, `--cols`, `--rows` | Convenience MRR shape options used when `--var` is not supplied. |
| `--max-utilization` | Enables Timeloop max-utilization behavior. |
| `--show-mapping` | Prints the Timeloop mapper loop text when available. |
| `--cache-results-dir` | Reads cached reconstructed Timeloop CSVs before running live Timeloop. |

## ROSA Application Commands

Cached report and validation:

```bash
conda run -n timeloop python optical_loop.py rosa --stage report
conda run -n timeloop python optical_loop.py rosa --stage validate
```

Regenerate lightweight ROSA artifacts:

```bash
conda run -n timeloop python optical_loop.py rosa --stage artifacts
```

Run the full ROSA architecture workflow through Timeloop:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage all --preset rosa-full
```

Run only hybrid mappings:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage hybrid --hybrid-family both
```

## Python API

From the repository root without installation, import `optical_loop` once to
bootstrap the local package name before importing from `opticalloop`.

Core usage:

```python
import optical_loop  # bootstraps local opticalloop package
from opticalloop import LayerSimulator, MRRMacroConfig, TimeloopLayerRef

layer = TimeloopLayerRef(network="alexnet", layer_path="alexnet/0")
architecture = MRRMacroConfig(
    n_tiles=1,
    n_pes=1,
    n_cols=100,
    n_rows=12,
    macro="proposed_mrr_optical_shift_add",
    max_utilization=False,
)
result = LayerSimulator(layer=layer, architecture=architecture).run()
```

Generic macro usage:

```python
import optical_loop  # bootstraps local opticalloop package
from opticalloop import TimeloopBackend, TimeloopLayerRef, TimeloopMacroConfig

layer = TimeloopLayerRef("deap_deepbench", "deap_deepbench/bench0")
architecture = TimeloopMacroConfig(
    macro="deap_cnns",
    system="fetch_all_lpddr4",
    variables={"N_COLUMNS": 100, "N_ROWS": 12, "N_Conv": 1},
    max_utilization=False,
)
result = TimeloopBackend().run_layer(layer, architecture)
print(result.mapping_text)
```

ROSA application usage:

```python
from opticalloop.applications.rosa import default_rosa_workflow

workflow = default_rosa_workflow()
report = workflow.cache_report()
```

Application-specific classes are exported from `opticalloop.applications.rosa`, not from the root `opticalloop` namespace.
