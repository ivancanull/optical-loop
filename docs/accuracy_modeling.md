# ONNSim accuracy modeling

The design-policy example evaluates one layer mapping policy against both
Timeloop-derived network EDP and MRR-aware whole-network accuracy. The native
Timeloop path supplies cycles, latency, energy, and area, while a localized
ResNet18/MRR subprocess supplies accuracy under the documented variation model.

## Interface

The canonical layer manifest maps each Timeloop workload ID to the corresponding
PyTorch module. A separate policy selects `WS` or `IS` and a temporal symbol
width for every layer. The same policy can therefore drive both Timeloop macro
selection and ONNSim variation placement.

The first supported manifest and policy are:

```text
config/accuracy/resnet18_cifar10_layers.yaml
config/accuracy/resnet18_hybrid_1bit.yaml
```

The pinned Docker image is authoritative for Timeloop/Accelergy performance
simulation. Accuracy evaluation uses a separate, optional PyTorch/CUDA Conda
environment named `optical-loop`. It is cloned from the native `timeloop`
environment and adds the packages listed in `environment-optical-loop.yml`.
The ready-to-run ResNet18 configuration is
`config/accuracy/onnsim_resnet18_cifar10_server.yaml`; its dataset path is
resolved from the accuracy subprocess working directory.

Create or refresh the optional accuracy environment from the repository root:

```bash
conda create -y -n optical-loop --clone timeloop
conda env update -n optical-loop -f environment-optical-loop.yml
conda activate optical-loop
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
conda activate optical-loop
```

The environment retains TimeloopFE, Accelergy, the native Timeloop binaries,
and CUDA PyTorch from `timeloop`, then adds torchvision, OmegaConf, regex, and
PrettyTable. The required ResNet18 and MRR runtime is maintained in OpticalLoop.

Run the optional accuracy backend after providing a quantized ResNet18
checkpoint and the CIFAR-10 dataset at the configured location:

```bash
python optical_loop.py accuracy \
  --layer-manifest config/accuracy/resnet18_cifar10_layers.yaml \
  --policy config/accuracy/resnet18_hybrid_1bit.yaml \
  --model-config config/accuracy/onnsim_resnet18_cifar10_server.yaml \
  --checkpoint reference/onnsim/checkpoints/best_model_quantized_resnet18_w8b.pth \
  --onnsim-root reference/onnsim \
  --runs 5 --seed 0 \
  --output results/accuracy/resnet18_hybrid.json
```

Accuracy is a whole-network measurement. When it is attached to layer/design
rows, `accuracy_status` is set to `MODELED_WHOLE_NETWORK`; repeated values must
not be summed or treated as independent layer costs.

## Online WS/IS mapping

Run the simple ResNet18 hill-climbing optimizer with live Timeloop EDP and
ONNSim accuracy:

```bash
conda run -n optical-loop python optical_loop.py optimize-mapping \
  --config config/optimization/resnet18_online.yaml \
  --output-dir results/optimization/resnet18
```

The optimizer evaluates the predefined policy, all-WS, all-IS, and deterministic
one- or two-layer mutations. It rejects mappings below the configured accuracy
floor and minimizes the normalized weighted EDP/accuracy cost. Timeloop choices
and accuracy evaluations are cached in the output directory, so rerunning the
same command resumes without repeating completed backend work.

The output directory contains `provenance.json`, `edp_lookup.json`,
`policy_cache.json`, `trials.csv`, `best_policy.yaml`, and `best_result.json`.
The provenance fingerprint binds cached rows to the architecture, manifest,
checkpoint, dataset content, noise parameters, source models, native mapper
and shared libraries, and Python/CUDA package versions. A changed fingerprint
requires a new output directory. Whole-network EDP is always calculated as
total layer energy multiplied by total layer latency.

## Slice-width thermal sensitivity

`thermal_std` is anchored to the configured 8-bit operating point. For a layer
using `b`-bit slices, OpticalLoop passes ONNSim the effective code-domain
standard deviation

```text
thermal_std(b) = thermal_std(8) * ((2^b - 1) / 255)^thermal_scaling_exponent
```

The default exponent is `0.5`. This phenomenological model represents the
greater sensitivity of narrowly spaced high-bit analog levels to a fixed
thermal perturbation while preserving the physical temperature variance. DAC
variation remains independent of slice width. Every accuracy result records
the parameters and effective values for 1/2/4/8-bit slices.

## Current boundary

The OpticalLoop subprocess runner resets Python, NumPy, CPU Torch, and CUDA
Torch random states for every requested run. Quantized ResNet18, CIFAR-10
preprocessing, and checkpoint-shape compatibility live inside OpticalLoop. The
subprocess provides PyTorch/GPU isolation. Current accuracy evidence covers the
1-bit input/hybrid semantics. Policies share the MB-OSA schema for 2/4/8-bit
widths, whose accuracy status remains `NOT_MODELED` pending matching temporal
slicing behavior in the inference backend.
