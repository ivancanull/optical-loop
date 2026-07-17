# WS/IS Multi-Bit Optical Shift-and-Add Study

This research workflow separates two independent architecture choices:

| Macro | Stationary operand | First-MRR sliced operand | Accumulation |
| --- | --- | --- | --- |
| `mrr_ws_no_osa` | Weight | Input | Electronic shift-and-add |
| `mrr_ws_osa` | Weight | Input | Optical delay line |
| `mrr_is_no_osa` | Input | Weight | Electronic shift-and-add |
| `mrr_is_osa` | Input | Weight | Optical delay line |

All four macros use `FRONT_MRR_SLICE_BITS={1,2,4,8}`. The total operand
precision remains 8 bits, so the number of temporal symbols is respectively
8, 4, 2, and 1. The physical MRR count and core shape do not change. At 8 bits
there is no temporal accumulation and both accumulator types are bypassed; this
is the Analog execution mode.

The no-OSA macros place a digital shift-and-add after conversion. Its Timeloop
energy is scaled from one access per temporal symbol to the effective
`N_TEMPORAL_SLICES - 1` additions. The OSA macros retain the paper delay-line
mapping: output dimension `X` for WS and `Y` for IS.

## Experiment matrix

The focused full matrix contains 42,240 native mapper jobs:

- WS-OSA and IS-OSA at 1/2/4/8 bits;
- WS-no-OSA and IS-no-OSA at 1 and 8 bits;
- 352 layers from six workloads and ten physical core shapes.

The 72-job smoke uses one representative layer per workload and all twelve
structure/width modes. Run the pinned environment with:

```bash
make multislice-smoke
WORKERS=16 make multislice-full
```

Direct commands are:

```bash
python3 optical_loop.py multislice doctor
python3 optical_loop.py multislice smoke --workers 8
python3 optical_loop.py multislice full --workers 16
python3 optical_loop.py multislice analyze --run-dir multislice-runs/<run-id>
python3 optical_loop.py multislice validate --run-dir multislice-runs/<run-id>
```

Runs are immutable and resumable by run ID. Repeat the same command after an
interruption. Do not copy checkpoint JSON between run IDs.

## Analysis

Generated artifacts report fixed WS/IS mappings, the paper-style 1-bit Mixed
mapping, ASWM-WS, ASWM-IS, and Joint-ASWM. Joint-ASWM selects one of the six
`(WS/IS, 1/2/4-bit)` choices per layer while keeping one physical core per
network. Selection uses the exact cumulative energy/latency Pareto frontier.

The primary DAC model is 5.2 pJ per represented bit. Reports also include a
constant-symbol optimistic bound, a `2^bits-1` Walden bound, and 0/0.5/1 dB
laser compensation per optical shift stage. These are model sensitivities, not
measured accuracy results.

Validation checks exact job coverage, true mapper stationarity, temporal slice
and accumulation counts, correct accumulator selection, DAC resolution,
constant MRR area/count, 8-bit bypass equivalence, units, frequency, EDP, and
adaptive-versus-fixed dominance.

Accuracy, training, noise, and thermal effects are explicitly `NOT_MODELED`.
No EDP result in this study is an accuracy claim.
