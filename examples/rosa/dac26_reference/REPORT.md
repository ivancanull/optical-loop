# DAC26 EDP Reproduction Report

Overall status: **PASS_WITH_PAPER_GAPS**

## Provenance

```json
{
  "accelergy": "0.4",
  "git_commit": "734c6d8c9a1aae5dc693007b7369b827d205883e",
  "git_dirty": false,
  "platform": "Linux-5.15.0-176-generic-x86_64-with-glibc2.35",
  "python": "3.10.12",
  "timeloop_mapper": "timeloop-mapper size=2035552 sha256=2ff90b894d3152e032d8fc4c0db3391c426aee28f004e2480bf70b8d111a3f33",
  "timeloopfe": "0.4"
}
```

## Headline metrics

```json
{
  "available": true,
  "winner": "paper_optimum_8x8",
  "optimized_vs_compact_reduction": 0.3190754326696389,
  "optimized_vs_deap_reduction": 0.4492457425164815,
  "osa_reduction": 0.22926520230030834
}
```

## Checks

| Severity | Status | Check | Detail |
|---|---|---|---|
| ERROR | PASS | job_coverage | 7040/7040 successful jobs |
| ERROR | PASS | no_duplicate_jobs | 7040 unique |
| ERROR | PASS | edp_equals_energy_times_latency | max relative error=0.000e+00 |
| ERROR | PASS | committed_reference_metrics | max relative error=0.000%; missing=0 |
| ERROR | PASS | feasible_winner | paper_optimum_8x8 |
| WARNING | WARN | paper:optimized_vs_compact_reduction | actual=31.9075%, paper=26.0000%, delta=5.91% |
| WARNING | WARN | paper:optimized_vs_deap_reduction | actual=44.9246%, paper=64.0000%, delta=19.08% |
| WARNING | WARN | paper:osa_reduction | actual=22.9265%, paper=29.0000%, delta=6.07% |
| WARNING | WARN | paper:optimized_ode | raw ODE configuration/data unavailable |
| WARNING | WARN | paper:hybrid_edp | paper table values are supplied without a complete accuracy-independent hybrid experiment definition |
