# MB-OSA and ASWM Experiment Report

Overall status: **PASS**

Accuracy status: **NOT_MODELED**. No accuracy constraint or claim is applied.

## Primary-model best results

- alexnet: ASWM c8_r32, EDP=0.0309149; equal within numerical tolerance versus same-core best fixed (0.000%); EDP change versus analog reference=+61.603%
- gpt2_medium: ASWM c8_r32, EDP=224.846; equal within numerical tolerance versus same-core best fixed (0.000%); EDP change versus analog reference=+62.220%
- mobilenet_v3: ASWM paper_optimum_8x8, EDP=0.00994275; equal within numerical tolerance versus same-core best fixed (-0.000%); EDP change versus analog reference=+32.625%
- resnet18: ASWM c8_r32, EDP=0.210341; equal within numerical tolerance versus same-core best fixed (0.000%); EDP change versus analog reference=+59.323%
- vgg16: ASWM c8_r32, EDP=17.4926; equal within numerical tolerance versus same-core best fixed (0.000%); EDP change versus analog reference=+67.532%
- vision_transformer: ASWM c8_r32, EDP=7.60006; equal within numerical tolerance versus same-core best fixed (0.000%); EDP change versus analog reference=+61.429%

**Finding:** ASWM does not strictly improve primary-model EDP over the best uniform slice width. It selects a uniform 4-bit mapping; therefore no adaptive-EDP improvement is claimed. Across all 432 core/workload/sensitivity comparisons, 0 show a strict adaptive reduction.

## Shared-core ranking

- #1 c8_r32: geometric-mean EDP=1.2747
- #2 c8_r16: geometric-mean EDP=1.5263
- #3 paper_optimum_8x8: geometric-mean EDP=2.05076
- #4 c4_r32: geometric-mean EDP=2.12252
- #5 c4_r16: geometric-mean EDP=2.19965
- #6 c4_r8: geometric-mean EDP=3.05317
- #7 c8_r4: geometric-mean EDP=3.32226
- #8 compact_4x4: geometric-mean EDP=5.07154

## DAC and optical-loss sensitivity

| DAC model | Loss per stage (dB) | Minimum EDP | Median EDP | Maximum EDP |
|---|---:|---:|---:|---:|
| conservative_walden | 0 | 0.0182612 | 8.90552 | 2662.76 |
| conservative_walden | 0.5 | 0.0182836 | 8.9064 | 2664.39 |
| conservative_walden | 1 | 0.0183154 | 8.90756 | 2666.7 |
| linear_bit | 0 | 0.00994275 | 4.2675 | 1596.25 |
| linear_bit | 0.5 | 0.00994432 | 4.2678 | 1596.37 |
| linear_bit | 1 | 0.00994608 | 4.26815 | 1596.5 |
| optimistic_constant_symbol | 0 | 0.00387125 | 2.77787 | 513.968 |
| optimistic_constant_symbol | 0.5 | 0.00387291 | 2.77817 | 514.089 |
| optimistic_constant_symbol | 1 | 0.00387478 | 2.77852 | 514.225 |

## Validation

| Severity | Status | Check | Detail |
|---|---|---|---|
| ERROR | PASS | job_coverage | 14080/14080 |
| ERROR | PASS | no_duplicate_jobs | unique=14080 |
| ERROR | PASS | exact_expected_jobs | missing=0, unexpected=0 |
| ERROR | PASS | no_failed_job_results | result_files=14080, failed=0 |
| ERROR | PASS | supported_slice_widths | [np.int64(1), np.int64(2), np.int64(4), np.int64(8)] |
| ERROR | PASS | temporal_slice_counts | expected 8/4/2/1 |
| ERROR | PASS | accuracy_not_modeled | NOT_MODELED |
| ERROR | PASS | workload_layer_completeness | {"alexnet": {"missing": 0, "unexpected": 0}, "gpt2_medium": {"missing": 0, "unexpected": 0}, "mobilenet_v3": {"missing": 0, "unexpected": 0}, "resnet18": {"missing": 0, "unexpected": 0}, "vgg16": {"missing": 0, "unexpected": 0}, "vision_transformer": {"missing": 0, "unexpected": 0}} |
| ERROR | PASS | architecture_constraints | invalid_candidates=[], shape_mismatches=0 |
| ERROR | PASS | unit_consistency | positive=True, max_latency_relative=0.000e+00 |
| ERROR | PASS | frequency_consistency | expected_hz=5000000000, max_relative=0.000e+00 |
| ERROR | PASS | native_dac_resolution_scaling | max_absolute_ratio_deviation=0.0225564 |
| ERROR | PASS | edp_formula | max=0.000e+00 |
| ERROR | PASS | aswm_no_worse_than_fixed | worse=[] |
