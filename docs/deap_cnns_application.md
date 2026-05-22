# DEAP-CNNs Application

DEAP-CNNs is an included OpticalLoop application for architecture-level modeling of a digital-electronic and analog-photonic convolutional unit. The application converts device and workload information from the DEAP-CNNs article into Timeloop/Accelergy model assets under `workspace/`.

## Commands

Report and validate the committed lightweight artifacts:

```bash
conda run -n timeloop python optical_loop.py deap-cnns --stage report
conda run -n timeloop python optical_loop.py deap-cnns --stage validate
```

Run the Timeloop-backed application workflow:

```bash
conda run -n timeloop python optical_loop.py deap-cnns --mode rerun --stage all --architecture mnist-default
```

Supported architecture presets are `mnist-default`, `edge-small`, and `edge-large`.

## Device Parameters

The DEAP-CNNs device constants are stored in `DeapDeviceSpec` and emitted to `examples/deap_cnns/device_parameters_deap_cnns.csv`.

| Parameter | Value |
| --- | --- |
| MRR precision | 7 bits |
| MRR self-coupling and loss | `r = a = 0.99` |
| Max wavelengths | 100 |
| Max modulators | 1024 |
| Waveguide geometry | 500 nm width, 220 nm thickness, 5 um bend radius |
| Wavelength range | 1.5-1.6 um |
| Propagation estimate | 100 MRRs, 10 um radius, about 21 ps |
| Throughput limits | balanced PD 25 GS/s, TIA 10 GS/s, MRR modulation 128 GS/s, DAC/ADC 5 GS/s |
| Output cycle | 200 ps |
| Device power | laser 100 mW, MRR 19.5 mW, DAC 26 mW, TIA 17 mW, ADC 76 mW |

Power values are encoded in the Timeloop/Accelergy assets using the existing photonic component-table convention where possible. The DEAP-CNNs ADC uses the shared ADC estimator with `ADC_ENERGY_SCALE` calibrated to 76 mW at the 200 ps output cycle. These values are not evaluated by a separate OpticalLoop energy simulator.

## Architecture Mapping

The `deap_cnns` macro maps the DEAP convolutional unit to an optical macro with laser sources, input modulation, photonic weight-bank MRRs, balanced photodiode/TIA readout, passive accumulation, ADC, and memory interface.

For the runnable presets:

| Preset | Kernel edge | Input channels | Wavelengths | Modulators |
| --- | ---: | ---: | ---: | ---: |
| `mnist-default` | 5 | 8 | 25 | 200 |
| `edge-small` | 3 | 113 | 9 | 1017 |
| `edge-large` | 10 | 10 | 100 | 1000 |

The article text also discusses an `R=10, D=12` edge case. OpticalLoop rejects that exact setting because it would require 1200 modulators, above the stated 1024-modulator hardware limit.

## Workloads

The included `deap_mnist` workload contains only the two optical convolution layers described by the application:

| Layer | Shape |
| --- | --- |
| `conv0` | 28x28x1 input to 24x24x8 output with 5x5 kernels |
| `conv1` | 24x24x8 input to 20x20x8 output with 5x5x8 kernels |

Pooling, fully connected layers, MNIST training, and accuracy simulation are outside this v1 Timeloop architecture path.

## Reference Boundary

`reference/DEAP-CNNs.pdf` is local source material only and is ignored by git. The repository commits derived parameters, model assets, docs, tests, and small CSV artifacts, not the PDF itself.
