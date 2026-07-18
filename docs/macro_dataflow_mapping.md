# Canonical MRR Dataflows

The research model has exactly four ROSA MRR macros. Stationarity and temporal
accumulation are independent choices rather than encoded in historical names.

| Macro | Stationarity | Optical order | Accumulator |
| --- | --- | --- | --- |
| `mrr_ws_no_osa` | Weight stationary | sliced input → 8-bit weight | Digital shift-add |
| `mrr_ws_osa` | Weight stationary | sliced input → 8-bit weight | Optical delay line |
| `mrr_is_no_osa` | Input stationary | sliced weight → 8-bit input | Digital shift-add |
| `mrr_is_osa` | Input stationary | sliced weight → 8-bit input | Optical delay line |

WS keeps weights at the weight DAC/MRR and uses the paper WS spatial constraints.
Its accumulator traverses output dimension `X`. IS keeps inputs at the input
DAC, streams sliced weights through the first MRR, and maps only output channel
`M` across photonic PEs so those PEs reuse the stationary input. Its accumulator
traverses `Y`. These properties must be visible in native mapper loop text.
All macros accept the same public variables:

```text
FRONT_MRR_SLICE_BITS = 1 | 2 | 4 | 8
FRONT_MRR_RADIX = 2^FRONT_MRR_SLICE_BITS
N_TEMPORAL_SLICES = 8 / FRONT_MRR_SLICE_BITS
N_TEMPORAL_ACCUMULATIONS = N_TEMPORAL_SLICES - 1
```

For WS, `FRONT_MRR_SLICE_BITS` controls the input DAC/MRR and weight remains
8-bit. For IS it controls the weight DAC/MRR and input remains 8-bit. MRR
instance counts and core dimensions never depend on the slice width.

At 8 bits the accumulator is disabled, giving the Analog mapping. At smaller
widths, OSA macros retain partial sums in the delay line. No-OSA macros convert
each partial result and use the digital shift-add component; its energy scale
accounts for `N_TEMPORAL_SLICES - 1` actual additions rather than charging the
initial load as an addition.

`deap_cnns` remains a separate example macro and is not one of these four ROSA
mapping choices.
