# Macro And Dataflow Mapping

This note records the macro-to-dataflow correspondence used by the vendored
CiMLoop-style examples in `workspace/`. It is meant to remove ambiguity when
reproducing application workflows from Timeloop results.

## How Timeloop Sees A Macro

`workspace/models/top.yaml.jinja2` composes one run from:

- `arch/1_macro/<macro>/arch.yaml`: hierarchy, component placement, and
  spatial/temporal mapping constraints.
- `arch/1_macro/<iso>/variables_iso.yaml`: technology and device parameters
  that are intended to stay comparable across a macro family.
- `arch/1_macro/<macro>/variables_free.yaml`: macro shape and knobs such as
  `N_TILES`, `N_PES`, `N_COLUMNS`, `N_ROWS`, `ADC_RESOLUTION`, and
  `VOLTAGE_DAC_RESOLUTION`.
- `workloads/<network>/<layer>.yaml`: Timeloop problem dimensions.

OpticalLoop forwards only the macro name and variables to
`workspace/scripts/utils.quick_run(...)` through `TimeloopBackend`; it does not
reinterpret latency, energy, or cycles locally.

## Problem Dimensions

The convolution problem base uses these dimensions:

| Dimension | Meaning |
| --- | --- |
| `N` | Batch |
| `C` | Input channels |
| `M` | Output channels |
| `P`, `Q` | Output spatial dimensions |
| `R`, `S` | Kernel spatial dimensions |
| `X` | Input precision bits |
| `Y` | Weight precision bits |
| `Z` | Output precision bits |
| `G` | Groups |

When reading the macro constraints, `maximize_dims` tells Timeloop which
dimensions a spatial or temporal level prefers to expose first.

## ROSA MRR Macro Family

| Macro | Dataflow role | Optical accumulation | Main use in OpticalLoop | Result postfix |
| --- | --- | --- | --- | --- |
| `proposed_mrr` | Regular MRR weight-stationary-style dataflow | No explicit OSA delay line | ROSA no-OSA baseline sweep | `_1bit_input` |
| `proposed_mrr_optical_shift_add` | Regular MRR weight-stationary-style dataflow | OSA delay line on output-bit accumulation | ROSA OSA sweep and OSA hybrid regular side | `_1bit_input_osa` |
| `proposed_mrr_wi_optical_shift_add` | WI operand-swapped hybrid dataflow | OSA delay line on weight-bit-oriented accumulation | ROSA OSA hybrid selected side | `hybrid_mapping` |
| `proposed_mrr_1bit_input_delay_line` | Legacy 1-bit-input regular dataflow | Electrical/photonic delay-line accumulation | Legacy hybrid regular side | legacy hybrid CSV names |
| `proposed_mrr_1bit_input_delay_line_wi` | Legacy 1-bit-input WI hybrid dataflow | Delay line with WI temporal accumulation | Legacy hybrid selected side | legacy hybrid CSV names |

Important naming detail: the ROSA `_1bit_input` sweep does not require a
separate `proposed_mrr_1bit_input` macro directory. It uses `proposed_mrr` with
`VOLTAGE_DAC_RESOLUTION=1`.

## Regular MRR Dataflow

`proposed_mrr` and `proposed_mrr_optical_shift_add` share the same core operand
placement:

- `glb` keeps `Inputs` and `Outputs`; weights are treated as the stationary
  operand by the default container constraints and local weight path.
- `tile` exposes `N_TILES`; `photonic_pe` exposes `N_PES`.
- `row` exposes `N_ROWS` and prefers `Y` then `M`, so rows map weight bits and
  output channels.
- `column` exposes `N_COLUMNS` and prefers `R,S` then `C`, so columns map kernel
  positions and input channels.
- `laser`, `input_dac`, and `input_mrr` carry the input-side optical signal.
- `weight_dac` and `weight_mrr` represent the weight-side modulation at the
  cross points.
- `photodiode_output_readout`, `TIA`, and `adc` form the output conversion path.

The only structural OSA difference is that
`proposed_mrr_optical_shift_add` inserts a `delay_line` before conversion. Its
temporal constraint prefers `X`, so the output-side accumulation is aligned with
input-bit slicing before ADC/OAC readout.

## WI Hybrid Dataflow

`proposed_mrr_wi_optical_shift_add` swaps the operand emphasis used for selected
layers in the hybrid mapping:

- `glb` keeps `Weights` and `Outputs`.
- `input_dac` is constrained with `factors_only: [M=-1]`, while
  `laser`, `weight_dac`, and `weight_mrr` carry the weight-side optical signal.
- `row` prefers `N`, so the row fanout is used for batch-side parallelism.
- `column` prefers `R,S`, keeping the kernel spatial mapping but dropping the
  regular macro's explicit `C` preference.
- `input_mrr` is the cross-point modulation component.
- The OSA `delay_line` prefers `Y`, so accumulation follows weight-bit slicing.

In workflow terms, this macro is not run for every layer by default. It is chosen
only when a layer's hybrid YAML value is `True`.

## Legacy Delay-Line Hybrid Dataflow

The legacy pair keeps a more explicit buffer/cache/router topology:

- `proposed_mrr_1bit_input_delay_line` adds `shared_router_group`, `router`,
  `input_buffer`, `output_buffer`, `weight_cache`, and `input_cache`.
- It uses `dac` and `mrr` components instead of the newer `dac_cache` and
  `tomrr` components.
- The regular legacy macro maps `row` to `Y,M`, maps `column` to `R,S,C`, and
  sets the delay-line temporal preference to `X`.
- `proposed_mrr_1bit_input_delay_line_wi` keeps the same topology but maps
  `row` to `M` and sets the delay-line temporal preference to `Y`.

Use this pair when reproducing older CiMLoop hybrid scripts. Use the OSA pair
for the current ROSA OSA hybrid workflow.

## Hybrid YAML Semantics

The hybrid mapping files live in `workspace/hybrid_mapping/`. For each layer:

- `False` means use the regular macro.
- `True` means use the hybrid/WI macro.

Current in-repo mappings select:

| Network | Hybrid layers |
| --- | --- |
| `alexnet` | `0`, `1`, `2`, `3` |
| `vgg16` | `00`, `01` |
| `resnet18` | `00`, `01`, `03`, `05`, `08` |
| `mobilenet_v3` | `00`, `04`, `07`, `10` |
| `mobilenet` | `00`, `features.0.pointwise`, `features.1.depthwise`, `features.2.depthwise` |

For OSA hybrid:

```text
False -> proposed_mrr_optical_shift_add
True  -> proposed_mrr_wi_optical_shift_add
```

For legacy delay-line hybrid:

```text
False -> proposed_mrr_1bit_input_delay_line
True  -> proposed_mrr_1bit_input_delay_line_wi
```

## DEAP-CNNs Notebook Macro

DEAP-CNNs is a notebook example that uses one canonical macro, `deap_cnns`.
It follows the same core row/column input-sharing logic as `proposed_mrr`,
with DEAP-CNNs names and device models, and is run through the generic
`layer --arch --workload --var` interface:

- `N_COLUMNS`, the wavelength-column count in one convolutional unit;
- `N_ROWS = D`, the fixed physical channel-line / weight-bank capacity;
- `N_Conv = nconv`, the number of parallel convolved output pixels;
- `conv_unit` spatially maps `P,Q` and then `N`;
- `laser`, `dac`, and `input_mrr` sit before the row/column crossing and have
  `N_COLUMNS` physical instances per convolution unit;
- `channel_weight_row` spatially maps only `C`; input-side sharing is expressed
  by placing the input components above this fanout, because Timeloop cannot map
  `C` while also applying the stronger `spatial_must_reuse_inputs` anchor;
- `wavelength_column` spatially maps `R,S` and uses `spatial_must_reuse_outputs`;
- `M/K` output filters are not mapped spatially and remain a sequential loop.

The notebook runs two settings:
`R=10,D=12` on `deap_deepbench/bench0`, so `N_COLUMNS=100` and `N_ROWS=12`;
`R=3,D=113` on `deap_deepbench/bench1`, so `N_COLUMNS=9` and `N_ROWS=113`.

The component placement follows the DEAP unit accounting:

| Component | Hardware count / role |
| --- | --- |
| `laser` | one source per wavelength column, i.e. `N_COLUMNS` per convolution unit, shared across channel rows |
| `dac` | one Wang-style R-2R DAC per wavelength column; raw component output is reported as-is |
| `input_mrr` | one input modulator MRR per wavelength column, placed before row fanout so rows share it |
| `weight_mrr` | one optical weight-bank crossing per `wavelength_column` x `channel_weight_row` site, i.e. `N_COLUMNS*D` |
| `TIA` and `photodiode_output_readout` | `D` readout lines per convolution unit, represented as `N_Conv * D` physical instances |
| `adc` | one column-readout ADC per convolution unit, represented as `N_Conv` physical instances |

Weights are treated as configured optical MRR states, so there is no separate
per-convolution weight-DAC action path. The notebook displays raw Timeloop
component rows and mapper loop text; it does not use a DEAP-specific workflow
or calibrated post-processing path.

## Reproduction Checklist

When reproducing a CSV, record these fields together:

| Field | Why it matters |
| --- | --- |
| `macro` | Selects the architecture/dataflow template. |
| `N_TILES`, `N_PES` or `N_Conv`, `N_COLUMNS`, `N_ROWS` | Defines the macro array shape or explicit generic Timeloop variables. |
| `VOLTAGE_DAC_RESOLUTION` | Distinguishes 1-bit input sweeps from higher-resolution input paths. |
| `system` | Selects the memory wrapper, usually `fetch_all_lpddr4`. |
| `max_utilization` | Changes mapper constraints when enabled. ROSA cached workflows use `False`. |
| `layer` | Must match the workload layer key used in the row index. |
| `hybrid_family` and YAML value | Required to know which macro ran for a hybrid layer. |

For ROSA headline reproduction, the safe defaults are:

```text
system = fetch_all_lpddr4
scaling = "aggressive"
VOLTAGE_DAC_RESOLUTION = 1
max_utilization = False
```
