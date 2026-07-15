# DEAP-CNNs Notebook Example

DEAP-CNNs is included as a small OpticalLoop example, not as a dedicated simulator interface. The example uses the canonical `deap_cnns` Timeloop macro directly with a workload name and explicit Timeloop variables.

Live runs remain Timeloop-only:

```text
TimeloopBackend -> workspace/scripts/utils.quick_run -> Timeloop mapper
```

There is no DEAP-specific Python workflow, validator, calibrated power model, or separate analytic runtime/energy simulator.

## Notebook

Open or execute:

```bash
examples/deap_cnns/deap_cnns_reproduction.ipynb
```

The notebook runs two cases:

| Case | Workload | Variables |
| --- | --- | --- |
| `R=10, D=12` | `deap_deepbench/bench0` | `N_COLUMNS=100`, `N_ROWS=12`, `N_Conv=1` |
| `R=3, D=113` | `deap_deepbench/bench1` | `N_COLUMNS=9`, `N_ROWS=113`, `N_Conv=1` |

It displays raw Timeloop metrics, raw per-component energy/power rows, simple device-count sanity checks, and `timeloop-mapper.map.txt` loop text through `SimulationResult.mapping_text`.

The current raw Timeloop numbers and mapper loop text are also recorded in `docs/deap_cnns_timeloop_data.md` for quick review without opening the notebook.

## CLI Equivalent

The same cases can be run from the generic layer command:

```bash
conda run -n timeloop python optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench0 \
  --var N_COLUMNS=100 \
  --var N_ROWS=12 \
  --var N_Conv=1 \
  --show-mapping

conda run -n timeloop python optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench1 \
  --var N_COLUMNS=9 \
  --var N_ROWS=113 \
  --var N_Conv=1 \
  --show-mapping
```

## Canonical Macro

`workspace/models/arch/1_macro/deap_cnns/` is the only DEAP-CNNs macro. It reuses the `proposed_mrr` row/column input-sharing logic with DEAP-CNNs names and device models:

```text
conv_unit(N_Conv)
  TIA
  photodiode_output_readout
  adc
  laser / dac / input_mrr  # N_COLUMNS physical input-side paths
  channel_weight_row(N_ROWS)  # reuses Inputs across rows
    wavelength_column(N_COLUMNS)
      weight_mrr
      *virtualized_mac
```

The main variables are:

| Variable | Meaning |
| --- | --- |
| `N_COLUMNS` | Wavelength-column count, treated as `R^2` in the notebook cases. |
| `N_ROWS` | Physical channel-line / weight-bank row count `D`. |
| `N_Conv` | Parallel convolution units. |

`TIA` and `adc` are inside `conv_unit`; total TIA count is `N_Conv * N_ROWS`, and total ADC count is `N_Conv`. Output filters remain temporal: `M/K` are not spatially mapped.

`laser`, `dac`, and `input_mrr` sit before the `channel_weight_row` fanout and each has `n_instances: N_COLUMNS`. This is how one input-side modulator path is shared across the `D` channel rows. `channel_weight_row` maps only `C`, so rows represent channel lines and unused rows stay idle when a workload has `C < D`. The stronger `spatial_must_reuse_inputs` anchor is not used because Timeloop treats `C` as an Inputs dimension and rejects mapping `C` while forbidding input-dataspace iteration. `wavelength_column` maps the kernel positions under each channel row.

The macro uses one logical `dac` node with `subclass: dac_r2r_ladder_compound`, following the same modeling style as the Wang-style R-2R DAC macros in the vendored workspace. Weight-bank values are treated as configured optical MRR states rather than a separate per-convolution weight-DAC action path.

## Workloads

Only the notebook workloads are included:

| Workload | Shape |
| --- | --- |
| `deap_deepbench/bench0` | `N=4, C=1, M=32, P=72, Q=349, R=20, S=5, stride=2` |
| `deap_deepbench/bench1` | `N=8, C=64, M=128, P=110, Q=110, R=3, S=3, stride=1` |

The hardware row count in the notebook is explicit. For example, `bench1` has workload `C=64` but uses the `R=3, D=113` hardware setting with `N_ROWS=113`.

## Boundary

`reference/` and `paper/` remain local source/context directories and are ignored by git. The repo commits derived Timeloop assets and the notebook example, not external PDFs or parent-repo result comparisons.
