# OpticalLoop

OpticalLoop is a **Timeloop-backed optical accelerator simulator**. Its goal is simple: keep optical computing architectures, workloads, component models, and run scripts in one self-contained repository, then use the Timeloop mapper to produce cycles, latency, energy, area, component breakdowns, and mapping text.

OpticalLoop **does not implement an additional analytic simulator** for energy, cycles, latency, area, or component power. All live simulation flows through:

```text
OpticalLoop CLI/Python API -> TimeloopBackend -> workspace/scripts/utils.quick_run -> Timeloop mapper
```

## Paper Reference

This repository accompanies the DAC 2026 manuscript:

> Huifan Zhang, Yun Hu, Caizhi Sheng, Yurui Qu, and Pingqiang Zhou. 2026.
> *ROSA: Robust and Energy-Efficient Microring-Based Optical Neural Networks via
> Optical Shift-and-Add and Layer-Wise Hybrid Mapping*. DAC 2026.

Find the publication record and available versions through
[Google Scholar](https://scholar.google.com/scholar?q=%22ROSA%3A+Robust+and+Energy-Efficient+Microring-Based+Optical+Neural+Networks+via+Optical+Shift-and-Add+and+Layer-Wise+Hybrid+Mapping%22).
See [`docs/dac26_reproduction.md`](docs/dac26_reproduction.md) for the
independent EDP reproduction procedure and its documented paper-alignment gaps.

## Features

OpticalLoop currently provides three main capabilities:

| Feature | Description |
| --- | --- |
| Generic layer simulation | Run one Timeloop mapper job from a macro, workload, and explicit variables. |
| ROSA application | Reproduce the ROSA/CIMLoop-style MRR optical computing workflow, including OSA/no-OSA comparisons, architecture ranking, validation, and plots. |
| DEAP-CNNs notebook example | Run the `deap_cnns` macro through the same generic Timeloop path and inspect raw metrics, component breakdowns, and mapper loop text. |

## Quick Start

For paper readers, the supported clean-checkout path is Docker:

```bash
make doctor
make smoke
# Complete 7,040-job experiment:
WORKERS=8 make full
```

See [docs/dac26_reproduction.md](docs/dac26_reproduction.md) for runtime,
resume, artifact, validation, and paper-gap details. The container pins the
complete native Timeloop/Accelergy environment by immutable image digest.

Use the existing `timeloop` conda environment for live simulation:

```bash
conda run -n timeloop python optical_loop.py --help
```

No editable package install is required. `optical_loop.py` bootstraps the local `opticalloop` package from this repository.
The Conda path is a developer convenience and is not the authoritative paper
reproduction environment.

Run tests:

```bash
python -m pytest -q
```

## Generic Layer Simulation

The most general entrypoint is `layer`. You can directly specify a macro, workload, and Timeloop variables:

```bash
conda run -n timeloop python optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench0 \
  --var N_COLUMNS=100 \
  --var N_ROWS=12 \
  --var N_Conv=1 \
  --show-mapping
```

`--show-mapping` prints the Timeloop mapper loop text, which is useful for checking the actual dataflow.

For ROSA/MRR-style macros, you can also use the shape shorthand:

```bash
conda run -n timeloop python optical_loop.py layer \
  --arch proposed_mrr_optical_shift_add \
  --workload alexnet/0 \
  --tiles 1 \
  --pes 1 \
  --cols 100 \
  --rows 12
```

Common options:

| Option | Meaning |
| --- | --- |
| `--arch` | Timeloop macro name, such as `deap_cnns` or `proposed_mrr_optical_shift_add`. |
| `--workload` | Workload path, such as `alexnet/0` or `deap_deepbench/bench0`. |
| `--var KEY=VALUE` | Variable passed directly to Timeloop. May be repeated. |
| `--tiles --pes --cols --rows` | Convenience shape options for MRR-style architectures. |
| `--system` | System wrapper. Defaults to `fetch_all_lpddr4`. |
| `--show-mapping` | Print Timeloop mapper text. |

## ROSA Reproduction

ROSA is the only formal application workflow in this repository. Its code lives in `applications/rosa/`.

Read the committed reference artifacts and print the headline report:

```bash
conda run -n timeloop python optical_loop.py rosa --stage report
```

Validate the committed artifacts:

```bash
conda run -n timeloop python optical_loop.py rosa --stage validate
```

Recompute the DAC26 EDP-only comparisons from all six committed workload
aggregates:

```bash
conda run -n timeloop python optical_loop.py rosa --stage paper-edp
```

The executable notebook is
`examples/rosa/dac26_edp_reproduction.ipynb`; its complete experiment manifest
is `examples/rosa/paper_edp_config.yaml`. The paper's robust aggregate includes
an undocumented `lambda`, so the notebook reports the directly reproducible
geometric mean alongside the published targets rather than fitting that value.

Current reference checks:

| Check | Expected |
| --- | --- |
| AlexNet OSA best | `T1,P1,C100,R12`, EDP `0.0810225695` |
| Six-network OSA winner | `T1, P32, C8, R4`, score `0.8529673580` |

Rerun the full ROSA Timeloop workflow:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage all --preset rosa-full
```

Run only the hybrid mapping workflow:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage hybrid --hybrid-family both
```

ROSA lightweight artifacts are stored in:

```text
examples/rosa/results/
examples/rosa/plots/
```

## DEAP-CNNs Example

DEAP-CNNs is **not a separate application workflow**. It is a notebook example showing how to run an optical computing macro directly with the generic `TimeloopMacroConfig` and `TimeloopBackend`.

Notebook:

```text
examples/deap_cnns/deap_cnns_reproduction.ipynb
```

It contains two cases:

| Case | Workload | Variables |
| --- | --- | --- |
| `R=10, D=12` | `deap_deepbench/bench0` | `N_COLUMNS=100`, `N_ROWS=12`, `N_Conv=1` |
| `R=3, D=113` | `deap_deepbench/bench1` | `N_COLUMNS=9`, `N_ROWS=113`, `N_Conv=1` |

Equivalent CLI commands:

```bash
conda run -n timeloop python optical_loop.py layer --arch deap_cnns \
  --workload deap_deepbench/bench0 --var N_COLUMNS=100 --var N_ROWS=12 --var N_Conv=1

conda run -n timeloop python optical_loop.py layer --arch deap_cnns \
  --workload deap_deepbench/bench1 --var N_COLUMNS=9 --var N_ROWS=113 --var N_Conv=1
```

Raw Timeloop metrics, component power/energy/area rows, and mapper loop text for these two cases are recorded in `docs/deap_cnns_timeloop_data.md`.

## Python API

Generic Timeloop run:

```python
import optical_loop  # bootstraps local opticalloop package
from opticalloop import TimeloopBackend, TimeloopLayerRef, TimeloopMacroConfig

layer = TimeloopLayerRef(
    network="deap_deepbench",
    layer_path="deap_deepbench/bench0",
)

architecture = TimeloopMacroConfig(
    macro="deap_cnns",
    system="fetch_all_lpddr4",
    variables={"N_COLUMNS": 100, "N_ROWS": 12, "N_Conv": 1},
    max_utilization=False,
)

result = TimeloopBackend().run_layer(layer, architecture)
print(result.energy_j, result.latency_s)
print(result.mapping_text)
```

MRR-style convenience config:

```python
import optical_loop
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

## Repo Structure

| Path | Description |
| --- | --- |
| `optical_loop.py` | CLI entrypoint. |
| `backend.py` | Timeloop adapter and the only live mapper call boundary. |
| `config/` | Macro and workload config dataclasses. |
| `result.py` | `SimulationResult`, including Timeloop stats and mapping text. |
| `simulator/` | Single-layer simulation facade. |
| `workflow/results.py` | CSV writing, reconstruction, and aggregation utilities. |
| `module_data.py` | Tidy module-level energy, area, and power rows. |
| `applications/rosa/` | ROSA application workflow. |
| `examples/` | ROSA artifacts and the DEAP-CNNs notebook example. |
| `workspace/` | Vendored Timeloop models, components, workloads, and scripts. |
| `docs/` | Detailed documentation. |
| `tests/` | Unit and integration-style checks. |

## Documentation Map

| Document | Purpose |
| --- | --- |
| `docs/simulator_overview.md` | Core simulator architecture and Timeloop boundary. |
| `docs/cli_and_api.md` | CLI and Python API usage. |
| `docs/rosa_application.md` | ROSA workflow, validation, and rerun commands. |
| `docs/deap_cnns_application.md` | DEAP-CNNs generic notebook example. |
| `docs/deap_cnns_timeloop_data.md` | Raw DEAP-CNNs Timeloop layer data and mapping text. |
| `docs/macro_dataflow_mapping.md` | Macro-to-dataflow correspondence for ROSA and DEAP-CNNs. |
| `docs/results_and_artifacts.md` | Result artifact meanings and validation formulas. |
| `docs/development_guidelines.md` | OOP, KISS, adapter-boundary, and cleanup rules. |
| `docs/dac26_reproduction.md` | Clean-checkout native simulation and validation guide. |

## Included Macros

The main optical macros live in `workspace/models/arch/1_macro/`:

| Macro | Purpose |
| --- | --- |
| `proposed_mrr` | ROSA no-OSA baseline. |
| `proposed_mrr_optical_shift_add` | ROSA OSA regular macro. |
| `proposed_mrr_wi_optical_shift_add` | ROSA OSA hybrid selected macro. |
| `proposed_mrr_1bit_input_delay_line` | Legacy delay-line regular macro. |
| `proposed_mrr_1bit_input_delay_line_wi` | Legacy delay-line hybrid macro. |
| `deap_cnns` | DEAP-CNNs notebook macro, reusing the `proposed_mrr` row/column input-sharing logic. |

## Correctness And Hygiene

Recommended checks after changes:

```bash
python -m pytest -q
python optical_loop.py rosa --stage report
python optical_loop.py rosa --stage validate
```

Optional live Timeloop smoke:

```bash
conda run -n timeloop python optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench0 \
  --var N_COLUMNS=100 \
  --var N_ROWS=12 \
  --var N_Conv=1 \
  --show-mapping
```

Portability check: scan README, docs, examples, applications, and workspace to ensure there are no machine-specific absolute paths or parent-result dependencies. CI/test scripts should follow the same hygiene rule.

## What Is Not Included

- No local analytic simulator for energy, latency, cycles, area, or component power.
- No PyTorch accuracy/noise/robustness experiments.
- No committed full `results/` tree.
- No committed `paper/` or `reference/` PDFs.

Generated local context such as `results/`, `paper/`, `reference/`, `temp/`, and `workspace/outputs/` is ignored by git.
