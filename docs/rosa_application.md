# ROSA Application

ROSA is the first OpticalLoop application preset. It provides an included architecture-level workflow for microring-resonator optical neural networks with optical shift-and-add and layer-wise hybrid mapping.

The application covers MRR-ONN architecture sweeps, no-OSA versus OSA comparison, six-network OSA ranking, and layer-wise hybrid mapping support. It does not include separate PyTorch DAC/thermal noise or CIFAR-10 accuracy simulations.

## What ROSA Runs Here

| Result area | Artifact support |
| --- | --- |
| Optimized MRR OPE sizing across CNN and Transformer workloads | Default ten-architecture sweep and six-network ranking over `alexnet`, `vgg16`, `resnet18`, `mobilenet_v3`, `gpt2_medium`, and `vision_transformer`. |
| No-OSA vs OSA architecture comparison | AlexNet aggregate CSVs and `examples/rosa/plots/alexnet_osa_edp_comparison.png`. |
| OSA architecture selection across workloads | `examples/rosa/results/aggregated_architecture_scores_1bit_input_osa_all_cached_networks.csv` and `examples/rosa/plots/six_network_osa_ranking.png`. |
| Layer-wise hybrid mapping workflow | Rerun-mode support for OSA and legacy delay-line hybrid macro families using `workspace/hybrid_mapping/*.yaml`. |
| Correctness checks | Schema checks, formula checks, headline value checks, and portable-path checks in `examples/rosa/results/validation_report.csv`. |

## Headline Checks

| Check | Expected result |
| --- | --- |
| AlexNet OSA best EDP | `T1,P1,C100,R12`, EDP `0.0810225695` |
| Six-network OSA ranking winner | `T1, P32, C8, R4`, score `0.8529673580` |

Run:

```bash
conda run -n timeloop python optical_loop.py rosa --stage report
conda run -n timeloop python optical_loop.py rosa --stage validate
```

## Full Rerun

Use rerun mode to regenerate live architecture metrics through Timeloop:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage all --preset rosa-full
```

This recreates Timeloop-backed energy, latency, area, power, and EDP artifacts. The full `results/` directory is ignored by git because it is larger and machine-generated.

## Hybrid Mapping

The ROSA application supports layer-wise hybrid mapping through boolean YAML files in `workspace/hybrid_mapping/`:

```bash
conda run -n timeloop python optical_loop.py rosa --mode rerun --stage hybrid --hybrid-family both
```

The `both` family runs the OSA hybrid macros and the legacy delay-line hybrid macros when mapping files exist for the selected networks.

## Self-Contained Data Boundary

ROSA validation reads only committed files under `examples/rosa/results/` or generated files under the in-repo `results/` directory. Full reruns call Timeloop through the vendored `workspace/` files in this repository.
