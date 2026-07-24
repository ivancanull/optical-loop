# CLI And API Usage

OpticalLoop can be used from the repository root without installation because `optical_loop.py` bootstraps the local `opticalloop` package.

## Paper reproduction CLI

Paper readers should use the pinned Docker workflow documented in
[`docs/rosa/dac26.md`](rosa/dac26.md). The underlying commands are:

```bash
docker compose run --rm opticalloop python3 optical_loop.py reproduce doctor
docker compose run --rm opticalloop python3 optical_loop.py reproduce smoke --workers 4
docker compose run --rm opticalloop python3 optical_loop.py reproduce full --workers 4
docker compose run --rm opticalloop python3 optical_loop.py reproduce analyze --run-dir reproduction-runs/<run-id>
docker compose run --rm opticalloop python3 optical_loop.py reproduce validate --run-dir reproduction-runs/<run-id>
```

`smoke` and `full` run native simulations and then analyze, validate, and
execute the notebook. Use `--skip-notebook` only for debugging. Successful jobs
resume by default. `--max-jobs N` converts `full` into a bounded batch of the next N pending jobs. It is the
recommended full-sweep mode. `--no-resume` rejects an existing provenance-keyed
run; use it only with an unused `--run-root`.

MB-OSA and ASWM use the parallel `multislice` application:

```bash
docker compose run --rm opticalloop python3 optical_loop.py multislice doctor
docker compose run --rm opticalloop python3 optical_loop.py multislice smoke --workers 4
docker compose run --rm opticalloop python3 optical_loop.py multislice full --workers 16
docker compose run --rm opticalloop python3 optical_loop.py multislice analyze --run-dir multislice-runs/<run-id>
```

## Generic Layer Command

Run one Timeloop-backed layer:

```bash
docker compose run --rm opticalloop python3 optical_loop.py layer \
  --workload alexnet/0 \
  --arch mrr_ws_osa \
  --tiles 1 --pes 1 --cols 100 --rows 12
```

Run an arbitrary macro with explicit Timeloop variables:

```bash
docker compose run --rm opticalloop python3 optical_loop.py layer \
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
    macro="mrr_ws_osa",
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

## Results And Artifacts

This section describes reusable result artifacts and the committed ROSA reference bundles.

### Generic Result Outputs

Application workflows can use `workflow/results.py` to write:

| Output | Meaning |
| --- | --- |
| `results_<save_name>_<network>_breakdown<postfix>.csv` | Per-layer Timeloop metrics and component breakdowns. |
| `results_<save_name>_<network>_combined<postfix>.csv` | Per-layer metrics with grouped component names when an application supplies groups. |
| `architecture_metrics_<save_name>_<network><postfix>.csv` | Architecture-level totals across layers. |
| `reconstructed/results_<network>_breakdown<postfix>.csv` | Parsed breakdown rows with network, layer, and architecture columns. |
| `reconstructed/aggregated_metrics_<network><postfix>.csv` | Aggregated EDP, TOPS, energy, area, cycles, latency, power, and efficiency metrics. |

### Committed Reference Bundles

| Bundle | Contents |
| --- | --- |
| `examples/rosa/dac26_reference/` | DAC26 aggregate metrics, figure, executed notebook, provenance, and validation report. |
| `examples/rosa/mb_osa_reference/` | Fixed-width and ASWM tables, figures, executed notebook, provenance, and validation report. |

Full job trees remain in ignored run directories. Reference bundles contain the compact evidence needed to inspect the released results.

### DEAP-CNNs Example

`examples/deap_cnns/deap_cnns_reproduction.ipynb` is a notebook example that runs two `deap_cnns` macro/workload cases through the generic backend. It displays raw Timeloop metrics, per-component rows, simple device-count sanity checks, and mapper loop text.

The DEAP-CNNs notebook and `docs/deap_cnns.md` carry the compact evidence; full Timeloop outputs remain under ignored runtime directories.

For convenience, `docs/deap_cnns.md` records the current raw Timeloop layer summaries, non-zero component breakdown rows, and mapper loop text for the two notebook cases. It records evidence from the generic layer workflow.

### Glossary

| Term | Meaning |
| --- | --- |
| ADC | Analog-to-digital converter after optical/electrical accumulation. |
| EDP | Energy-delay product, combining energy and latency. |
| IS | Input-stationary mapping. |
| MRR | Microring resonator. |
| OAC | Optical-to-analog conversion stage before digitization. |
| OPE | Optical processing element. |
| OSA | Optical shift-and-add. |
| WDM | Wavelength-division multiplexing. |
| WS | Weight-stationary mapping. |
