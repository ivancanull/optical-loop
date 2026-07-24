"""Self-contained ResNet18 runtime matching the pinned PlaNetZoo model.

The layer names, parameters, quantization, CIFAR-10 preprocessing, and
checkpoint loading semantics intentionally match PlaNetZoo commit
46d4bfba7e107aed794c2c68b5782ff4eb533028. The original code is MIT licensed;
see ``accuracy/PLANETZOO_LICENSE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class WeightQuantizer(torch.autograd.Function):
    """PlaNetZoo's signed/unsigned straight-through weight quantizer."""

    @staticmethod
    def forward(ctx, value, bits, scale, signed=True):
        qmin = -(2 ** (bits - 1)) if signed else 0
        qmax = 2 ** (bits - 1) - 1 if signed else 2**bits - 1
        return torch.clamp(torch.round(value / scale), qmin, qmax) * scale

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None, None, None


class QuantizedConv2d(nn.Module):
    """Weight-quantized convolution with checkpoint-compatible parameters."""

    def __init__(
        self, in_channels, out_channels, kernel_size, *, stride=1, padding=0,
        bias=True, weight_bits=8, signed=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = (1, 1)
        self.groups = 1
        self.weight_bits = weight_bits
        self.signed = signed
        self.freeze = False
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.weight_scale = nn.Parameter(torch.ones(1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    @staticmethod
    def _pair(value):
        return value if isinstance(value, tuple) else (value, value)

    def forward(self, value):
        weight = self.weight if self.freeze else WeightQuantizer.apply(
            self.weight, self.weight_bits, self.weight_scale, self.signed
        )
        return F.conv2d(
            value, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )


class QuantizedLinear(nn.Module):
    """Weight-quantized linear layer with checkpoint-compatible parameters."""

    def __init__(
        self, in_features, out_features, *, bias=True, weight_bits=8, signed=True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.signed = signed
        self.freeze = False
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_scale = nn.Parameter(torch.ones(1))
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, value):
        weight = self.weight if self.freeze else WeightQuantizer.apply(
            self.weight, self.weight_bits, self.weight_scale, self.signed
        )
        return F.linear(value, weight, self.bias)


class QuantizedBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, *,
        weight_bits=8, signed=True,
    ) -> None:
        super().__init__()
        self.conv1 = QuantizedConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False,
            weight_bits=weight_bits, signed=signed,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuantizedConv2d(
            out_channels, out_channels, 3, padding=1, bias=False,
            weight_bits=weight_bits, signed=signed,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, value):
        identity = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(value + identity)


class QuantizedResNet18(nn.Module):
    """Local ResNet18 with the exact PlaNetZoo module hierarchy."""

    def __init__(
        self, *, num_classes=10, input_channels=3,
        hidden_channels=(64, 128, 256, 512), dropout=0.0,
        weight_bits=8, signed=True,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != 4:
            raise ValueError("hidden_channels must contain exactly 4 values")
        self.weight_bits = weight_bits
        self.signed = signed
        self.in_channels = hidden_channels[0]
        self.conv1 = QuantizedConv2d(
            input_channels, hidden_channels[0], 7, stride=2, padding=3,
            bias=False, weight_bits=weight_bits, signed=signed,
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(hidden_channels[0], 2, 1)
        self.layer2 = self._make_layer(hidden_channels[1], 2, 2)
        self.layer3 = self._make_layer(hidden_channels[2], 2, 2)
        self.layer4 = self._make_layer(hidden_channels[3], 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = QuantizedLinear(
            hidden_channels[3], num_classes, weight_bits=weight_bits, signed=signed,
        )
        self.classifier = (
            nn.Sequential(nn.Dropout(dropout), classifier) if dropout > 0 else classifier
        )
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                QuantizedConv2d(
                    self.in_channels, out_channels, 1, stride=stride, bias=False,
                    weight_bits=self.weight_bits, signed=self.signed,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [QuantizedBasicBlock(
            self.in_channels, out_channels, stride, downsample,
            weight_bits=self.weight_bits, signed=self.signed,
        )]
        self.in_channels = out_channels
        layers.extend(
            QuantizedBasicBlock(
                self.in_channels, out_channels,
                weight_bits=self.weight_bits, signed=self.signed,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (QuantizedConv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (QuantizedLinear, nn.Linear)):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, value):
        value = self.maxpool(self.relu(self.bn1(self.conv1(value))))
        value = self.layer4(self.layer3(self.layer2(self.layer1(value))))
        return self.classifier(torch.flatten(self.avgpool(value), 1))


@dataclass(frozen=True)
class LocalModelConfig:
    input_channels: int = 3
    num_classes: int = 10
    model_type: str = "quantized_resnet18"
    weight_bits: int = 8
    dropout: float = 0.0
    signed: bool = True
    hidden_channels: Sequence[int] = (64, 128, 256, 512)

    def build(self) -> QuantizedResNet18:
        if self.model_type != "quantized_resnet18":
            raise ValueError("local accuracy runtime supports only quantized_resnet18")
        return QuantizedResNet18(
            input_channels=self.input_channels, num_classes=self.num_classes,
            weight_bits=self.weight_bits, dropout=self.dropout,
            signed=self.signed, hidden_channels=self.hidden_channels,
        )


class Cifar10Loader:
    """Construct the same deterministic test loader used by PlaNetZoo."""

    @staticmethod
    def test_loader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_dir), train=False, download=False, transform=transform,
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )


def load_checkpoint(model: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    """Load old PlaNetZoo checkpoints, retaining its scalar shape repair."""
    model_state = model.state_dict()
    state = dict(state)
    for key, value in state.items():
        if key not in model_state or "weight_scale" not in key:
            continue
        if value.shape == torch.Size([]) and model_state[key].shape == torch.Size([1]):
            state[key] = value.reshape(1)
        elif value.shape == torch.Size([1]) and model_state[key].shape == torch.Size([]):
            state[key] = value.squeeze()
    model.load_state_dict(state)
    return model


def apply_mrr_weight_variations(model, devices, layer_names):
    """Local copy of the ONNSim operation used for mapped weights."""
    selected = set(layer_names)
    for name, module in model.named_modules():
        if name in selected and hasattr(module, "weight_scale") and hasattr(module, "weight"):
            with torch.no_grad():
                module.weight.copy_(devices[name].simulate_variation(module.weight))
    return model


def apply_mrr_input_noise(model, devices, layer_names):
    """Local copy of the ONNSim operation used for mapped inputs."""
    selected = set(layer_names)
    for name, module in model.named_modules():
        if name not in selected or not (
            hasattr(module, "weight_scale") and hasattr(module, "weight")
        ):
            continue
        original_forward = module.forward

        def noisy_forward(value, original_forward=original_forward, name=name):
            devices[name].set_weight_mapping_range(
                0, 3.0, 1537, value.min().item(), value.max().item()
            )
            return original_forward(devices[name].simulate_variation(value))

        module.forward = noisy_forward
    return model
