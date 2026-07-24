# DEAP-CNNs Notebook Example

The DEAP-CNNs physical-fabric design example shows how array geometry, device
placement, and converter fanout become an executable Timeloop hierarchy. The
canonical `deap_cnns` macro is driven by a workload name and explicit hardware
variables.

Live runs remain Timeloop-only:

```text
TimeloopBackend -> workspace/scripts/utils.quick_run -> Timeloop mapper
```

The example intentionally reuses the generic `layer` command, Timeloop backend,
and native result schema. Its evidence consists of the macro definition, two
explicit workloads, component actions, and mapper loop text.

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


## CLI Equivalent

The same cases can be run from the generic layer command:

```bash
docker compose run --rm opticalloop python3 optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench0 \
  --var N_COLUMNS=100 \
  --var N_ROWS=12 \
  --var N_Conv=1 \
  --show-mapping

docker compose run --rm opticalloop python3 optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench1 \
  --var N_COLUMNS=9 \
  --var N_ROWS=113 \
  --var N_Conv=1 \
  --show-mapping
```

## Canonical Macro

`workspace/models/arch/1_macro/deap_cnns/` is the only DEAP-CNNs macro. It uses the same row/column optical modeling framework as the WS macros, with DEAP-CNNs names and device models:

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

`reference/` and `paper/` remain local source/context directories covered by
the ignore rules. The public repository contains the derived Timeloop assets,
the notebook example, and links to publication records.
## Recorded Timeloop Data

This page records raw Timeloop results for the current `deap_cnns` notebook
cases. The generic layer command and canonical `deap_cnns` macro produce these
numbers directly through the shared Timeloop backend.

## Commands

```bash
docker compose run --rm opticalloop python3 optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench0 \
  --var N_COLUMNS=100 \
  --var N_ROWS=12 \
  --var N_Conv=1 \
  --show-mapping

docker compose run --rm opticalloop python3 optical_loop.py layer \
  --arch deap_cnns \
  --workload deap_deepbench/bench1 \
  --var N_COLUMNS=9 \
  --var N_ROWS=113 \
  --var N_Conv=1 \
  --show-mapping
```

## Layer Summary

| Case | Workload | Variables | Cycles | Latency (s) | Energy (J) | Area (mm^2) | Avg. Power (W) | EDP | TOPS/W |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R=10,D=12` | `deap_deepbench/bench0` | `N_COLUMNS=100`, `N_ROWS=12`, `N_Conv=1` | 3,216,384 | 6.432768e-04 | 1.164280847020e-01 | 3.788389505660e-05 | 180.992202 | 7.489548575723e-05 | 0.005525 |
| `R=3,D=113` | `deap_deepbench/bench1` | `N_COLUMNS=9`, `N_ROWS=113`, `N_Conv=1` | 12,390,400 | 2.478080e-03 | 7.985455758297e-01 | 2.789029960700e-05 | 322.243663 | 1.978859820552e-03 | 0.017875 |

`Avg. Power` is `energy_j / latency_s` from the same Timeloop result.

## Component Breakdown: `bench0`, `R=10,D=12`

Non-zero component rows are shown below. Timeloop also emits zero-energy connector rows such as `laser <==> dac`; those rows are omitted here for readability.

| Component | Energy (J) | Power (W) | Area (mm^2) |
| --- | ---: | ---: | ---: |
| `laser` | 6.817897820160e-02 | 105.987000 | 1.046010000000e-05 |
| `main_memory` | 1.842376919040e-02 | 28.640500 | 0.000000000000e+00 |
| `input_mrr` | 1.334895851520e-02 | 20.751500 | 6.800000000000e-07 |
| `weight_mrr` | 1.334895851520e-02 | 20.751500 | 8.160000000000e-06 |
| `TIA` | 1.390056837120e-03 | 2.160900 | 1.632000000000e-07 |
| `adc` | 1.037026529280e-03 | 1.612100 | 1.694560000000e-05 |
| `dac` | 6.862026608640e-04 | 1.066730 | 1.061250000000e-07 |
| `photodiode_output_readout` | 1.191477288960e-05 | 0.018522 | 1.315800000000e-08 |
| `glb` | 2.217471621120e-06 | 0.003447 | 1.355650000000e-06 |
| `output_buffer` | 2.007859875840e-09 | 0.000003 | 6.205660000000e-11 |

## Component Breakdown: `bench1`, `R=3,D=113`

| Component | Energy (J) | Power (W) | Area (mm^2) |
| --- | ---: | ---: | ---: |
| `main_memory` | 4.014632337408e-01 | 162.005760 | 0.000000000000e+00 |
| `weight_mrr` | 2.962015322112e-01 | 119.528640 | 6.915600000000e-06 |
| `TIA` | 5.042769887232e-02 | 20.349504 | 1.536800000000e-06 |
| `laser` | 2.364716364595e-02 | 9.542534 | 9.414090000000e-07 |
| `dac` | 1.522622752358e-02 | 6.144365 | 9.551240000000e-09 |
| `input_mrr` | 4.602139508736e-03 | 1.857139 | 6.120000000000e-08 |
| `adc` | 3.990152871936e-03 | 1.610179 | 1.694560000000e-05 |
| `photodiode_output_readout` | 2.937535856640e-03 | 1.185408 | 1.239050000000e-07 |
| `glb` | 4.797975232512e-05 | 0.019362 | 1.355650000000e-06 |
| `output_buffer` | 1.911846253363e-06 | 0.000772 | 5.843670000000e-10 |

## Mapper Loop Text: `bench0`

```text
main_memory [ Weights:22400 (22400) Inputs:3179736 (3179736) Outputs:22514688 (22514688) ]
------------------------------------------------------------------------------------------
| for N in [0:4)
|   for P in [0:4)

glb [ Inputs:264978 (264978) Outputs:1407168 (1407168) ]
--------------------------------------------------------
|     for P in [0:18)
|       for Q in [0:349)
|         for M in [0:32)

output_buffer [ Outputs:7 (7) ]
TIA [ Outputs:7 (7) ]
photodiode_output_readout [ Outputs:7 (7) ]
adc [ Outputs:7 (7) ]
laser [ Inputs:700 (700) ]
dac [ Inputs:700 (700) ]
input_mrr [ Inputs:700 (700) ]
inter_channel_weight_row_spatial [ ]
inter_wavelength_column_spatial [ ]
-----------------------------------
|           for S in [0:5) (Spatial-X)
|             for R in [0:20) (Spatial-X)

weight_mrr [ Weights:7 (7) Inputs:7 (7) ]
inter_1bit_x_1bit_mac_spatial [ ]
---------------------------------
|               for Z in [0:7) (Spatial-X)
|                 for Y in [0:7) (Spatial-X)
|                   for X in [0:7) (Spatial-X)

here_to_fix_a_bug [ ]
---------------------
|                     << Compute >>
```

## Mapper Loop Text: `bench1`

```text
main_memory [ Weights:516096 (516096) Inputs:44957696 (44957696) Outputs:86732800 (86732800) ]
----------------------------------------------------------------------------------------------
| for N in [0:4)
|   for P in [0:110)

glb [ Inputs:301056 (301056) Outputs:197120 (197120) ]
------------------------------------------------------
|     for Q in [0:110)
|       for M in [0:128)
|         for N in [0:2)

output_buffer [ Outputs:7 (7) ]
TIA [ Outputs:7 (7) ]
photodiode_output_readout [ Outputs:7 (7) ]
adc [ Outputs:7 (7) ]
laser [ Inputs:4032 (4032) ]
dac [ Inputs:4032 (4032) ]
input_mrr [ Inputs:4032 (4032) ]
inter_channel_weight_row_spatial [ ]
------------------------------------
|           for C in [0:64) (Spatial-Y)

inter_wavelength_column_spatial [ ]
-----------------------------------
|             for S in [0:3) (Spatial-X)
|               for R in [0:3) (Spatial-X)

weight_mrr [ Weights:7 (7) Inputs:7 (7) ]
inter_1bit_x_1bit_mac_spatial [ ]
---------------------------------
|                 for Z in [0:7) (Spatial-X)
|                   for Y in [0:7) (Spatial-X)
|                     for X in [0:7) (Spatial-X)

here_to_fix_a_bug [ ]
---------------------
|                       << Compute >>
```

## Interpretation Notes

- The data above is raw Timeloop output after the current macro/dataflow cleanup.
- `main_memory` is part of the Timeloop system wrapper and is intentionally kept in the raw result. It is separate from the optical macro components.
- `input_mrr` sits before the channel-row fanout, so it represents the shared input modulation path. `weight_mrr` is at the row/column crossing.
- `TIA` and `adc` are inside `conv_unit`; total TIA count is `N_Conv * N_ROWS`, and total ADC count is `N_Conv`.
- `M/K` output filters remain temporal rather than spatially mapped.
