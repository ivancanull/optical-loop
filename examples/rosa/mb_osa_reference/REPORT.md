# MB-OSA and ASWM Experiment Report

Overall status: **PASS**

Run tier: **FULL**.

Accuracy status: **NOT_MODELED**. No accuracy constraint or claim is applied.

## Environment provenance

- Run ID: `mb-osa-aswm-full-9d46611425c6fdee`
- Manifest SHA-256: `074c6d15f742a1db9c559757d3053249b561a8819476aeaea6a30cde57d9b763`
- Source commit: `451631f22bd449c693af95c256599a71346d4e74`
- Timeloop mapper: timeloop-mapper size=2035552 sha256=2ff90b894d3152e032d8fc4c0db3391c426aee28f004e2480bf70b8d111a3f33
- TimeloopFE: 0.4
- Accelergy: 0.4
- Python: 3.10.12
- Jobs: 42240 successful, 0 failed, 0 remaining
- Created: 2026-07-19T16:49:03.107413+00:00
- Updated: 2026-07-20T01:54:04.727283+00:00

## Primary-model best results

- ASWM-IS/alexnet: c8_r32, EDP=0.0421984; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=-45.828%
- ASWM-IS/gpt2_medium: c8_r32, EDP=370.456; strict reduction versus same-core best fixed (0.000028%); EDP change versus analog reference=-75.669%
- ASWM-IS/mobilenet_v3: c8_r32, EDP=0.00734855; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=-49.382%
- ASWM-IS/resnet18: c8_r32, EDP=0.326034; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=-36.794%
- ASWM-IS/vgg16: c8_r32, EDP=29.2564; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=-30.912%
- ASWM-IS/vision_transformer: c8_r32, EDP=11.6504; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=-77.633%
- ASWM-WS/alexnet: c8_r32, EDP=0.0312659; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+61.952%
- ASWM-WS/gpt2_medium: c8_r32, EDP=227.318; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=+62.554%
- ASWM-WS/mobilenet_v3: paper_optimum_8x8, EDP=0.0100305; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+33.016%
- ASWM-WS/resnet18: c8_r32, EDP=0.214113; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+61.147%
- ASWM-WS/vgg16: c8_r32, EDP=18.0857; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=+72.113%
- ASWM-WS/vision_transformer: c8_r32, EDP=7.68309; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+61.766%
- Joint-ASWM/alexnet: c8_r32, EDP=0.0312659; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+61.952%
- Joint-ASWM/gpt2_medium: c8_r32, EDP=227.318; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=+62.554%
- Joint-ASWM/mobilenet_v3: c8_r32, EDP=0.00384582; strict reduction versus same-core best fixed (47.665612%); EDP change versus analog reference=-72.672%
- Joint-ASWM/resnet18: c8_r32, EDP=0.214113; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+61.147%
- Joint-ASWM/vgg16: c8_r32, EDP=18.0857; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=+72.113%
- Joint-ASWM/vision_transformer: c8_r32, EDP=7.68309; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+61.766%
- Mixed-1bit/alexnet: c8_r32, EDP=0.17723; strict reduction versus same-core best fixed (10.188487%); EDP change versus analog reference=+818.022%
- Mixed-1bit/gpt2_medium: c8_r32, EDP=1532.79; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+996.085%
- Mixed-1bit/mobilenet_v3: c8_r32, EDP=0.0208785; strict reduction versus same-core best fixed (32.883177%); EDP change versus analog reference=+48.360%
- Mixed-1bit/resnet18: c4_r16, EDP=1.22379; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=+392.708%
- Mixed-1bit/vgg16: c4_r16, EDP=90.4941; equal within numerical tolerance versus same-core best fixed (0.000000%); EDP change versus analog reference=+399.523%
- Mixed-1bit/vision_transformer: c8_r32, EDP=52.3701; equal within numerical tolerance versus same-core best fixed (-0.000000%); EDP change versus analog reference=+1002.642%

## Shared-core ranking

- ASWM-IS #1 c8_r32: geometric-mean EDP=1.52877
- ASWM-IS #2 c4_r32: geometric-mean EDP=2.85659
- ASWM-IS #3 c8_r16: geometric-mean EDP=3.2215
- ASWM-IS #4 c4_r16: geometric-mean EDP=6.20575
- ASWM-IS #5 paper_optimum_8x8: geometric-mean EDP=6.90837
- ASWM-IS #6 c4_r8: geometric-mean EDP=17.6206
- ASWM-IS #7 c8_r4: geometric-mean EDP=18.2614
- ASWM-IS #8 compact_4x4: geometric-mean EDP=44.3507
- ASWM-WS #1 c8_r32: geometric-mean EDP=1.29364
- ASWM-WS #2 c8_r16: geometric-mean EDP=1.55948
- ASWM-WS #3 paper_optimum_8x8: geometric-mean EDP=2.10103
- ASWM-WS #4 c4_r32: geometric-mean EDP=2.15736
- ASWM-WS #5 c4_r16: geometric-mean EDP=2.22476
- ASWM-WS #6 c4_r8: geometric-mean EDP=3.09392
- ASWM-WS #7 c8_r4: geometric-mean EDP=3.40521
- ASWM-WS #8 compact_4x4: geometric-mean EDP=5.13816
- Joint-ASWM #1 c8_r32: geometric-mean EDP=0.966129
- Joint-ASWM #2 c8_r16: geometric-mean EDP=1.37947
- Joint-ASWM #3 c4_r32: geometric-mean EDP=1.67562
- Joint-ASWM #4 paper_optimum_8x8: geometric-mean EDP=2.10103
- Joint-ASWM #5 c4_r16: geometric-mean EDP=2.13008
- Joint-ASWM #6 c4_r8: geometric-mean EDP=3.09392
- Joint-ASWM #7 c8_r4: geometric-mean EDP=3.40521
- Joint-ASWM #8 compact_4x4: geometric-mean EDP=5.13816
- Mixed-1bit #1 c8_r32: geometric-mean EDP=5.78881
- Mixed-1bit #2 c8_r16: geometric-mean EDP=6.67069
- Mixed-1bit #3 paper_optimum_8x8: geometric-mean EDP=7.40658
- Mixed-1bit #4 c4_r32: geometric-mean EDP=7.76328
- Mixed-1bit #5 c4_r16: geometric-mean EDP=7.84144
- Mixed-1bit #6 c4_r8: geometric-mean EDP=8.10147
- Mixed-1bit #7 c8_r4: geometric-mean EDP=9.13506
- Mixed-1bit #8 compact_4x4: geometric-mean EDP=10.3632

## DAC and optical-loss sensitivity

| DAC model | Loss per stage (dB) | Minimum EDP | Median EDP | Maximum EDP |
|---|---:|---:|---:|---:|
| conservative_walden | 0 | 0.00659539 | 13.9745 | 9967.58 |
| conservative_walden | 0.5 | 0.00659555 | 13.9748 | 9967.64 |
| conservative_walden | 1 | 0.00659573 | 13.9752 | 9967.71 |
| linear_bit | 0 | 0.00384582 | 7.54356 | 6015.21 |
| linear_bit | 0.5 | 0.00384598 | 7.54386 | 6015.27 |
| linear_bit | 1 | 0.00384617 | 7.54421 | 6015.34 |
| optimistic_constant_symbol | 0 | 0.00309594 | 5.32635 | 4937.29 |
| optimistic_constant_symbol | 0.5 | 0.0030961 | 5.32688 | 4937.35 |
| optimistic_constant_symbol | 1 | 0.00309628 | 5.32747 | 4937.42 |

## Validation

| Severity | Status | Check | Detail |
|---|---|---|---|
| ERROR | PASS | job_coverage | 42240/42240 |
| ERROR | PASS | no_duplicate_jobs | unique=42240 |
| ERROR | PASS | exact_expected_jobs | missing=0, unexpected=0 |
| ERROR | PASS | no_failed_job_results | result_files=42240, failed=0 |
| ERROR | PASS | supported_slice_widths | [np.int64(1), np.int64(2), np.int64(4), np.int64(8)] |
| ERROR | PASS | temporal_slice_counts | expected 8/4/2/1 |
| ERROR | PASS | temporal_accumulation_counts | expected 7/3/1/0 |
| ERROR | PASS | accuracy_not_modeled | NOT_MODELED |
| ERROR | PASS | accumulation_component_selection | optical=delay_line, digital=digital_shift_add, 8bit=bypass |
| ERROR | PASS | native_mapping_stationarity | WS accumulator traverses X; IS accumulator traverses Y and PE spatial loops only traverse M |
| ERROR | PASS | workload_layer_completeness | {"alexnet": {"missing": 0, "unexpected": 0}, "gpt2_medium": {"missing": 0, "unexpected": 0}, "mobilenet_v3": {"missing": 0, "unexpected": 0}, "resnet18": {"missing": 0, "unexpected": 0}, "vgg16": {"missing": 0, "unexpected": 0}, "vision_transformer": {"missing": 0, "unexpected": 0}} |
| ERROR | PASS | architecture_constraints | invalid_candidates=[], shape_mismatches=0 |
| ERROR | PASS | unit_consistency | positive=True, max_latency_relative=0.000e+00 |
| ERROR | PASS | frequency_consistency | expected_hz=5000000000, max_relative=0.000e+00 |
| ERROR | PASS | native_dac_resolution_distinct | strictly_increasing=True |
| ERROR | PASS | physical_mrr_area_invariant | max_spread_mm2=0.000e+00 |
| ERROR | PASS | analog_8bit_bypass_equivalence | cycles_equal=True, max_energy_relative=0.000e+00 |
| ERROR | PASS | edp_formula | max=0.000e+00 |
| ERROR | PASS | aswm_no_worse_than_fixed | worse=[] |
