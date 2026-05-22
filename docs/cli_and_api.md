# CLI And API Usage

OpticalLoop can be used from the repository root without installation because `optical_loop.py` bootstraps the local `opticalloop` package.

## Generic Layer Command

Run one Timeloop-backed layer:

```bash
conda run -n timeloop python optical_loop.py layer \
  --network alexnet \
  --layer alexnet/0 \
  --macro proposed_mrr_optical_shift_add \
  --tiles 1 --pes 1 --cols 100 --rows 12
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--system` | Timeloop system template name, default `fetch_all_lpddr4`. |
| `--voltage-dac-resolution` | Forwarded to Timeloop as `VOLTAGE_DAC_RESOLUTION`. |
| `--scaling` | Forwarded to Timeloop as `SCALING`. |
| `--max-utilization` | Enables Timeloop max-utilization behavior. |
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

Core usage:

```python
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

ROSA application usage:

```python
from opticalloop.applications.rosa import default_rosa_workflow

workflow = default_rosa_workflow()
report = workflow.cache_report()
```

Application-specific classes are exported from `opticalloop.applications.rosa`, not from the root `opticalloop` namespace.
