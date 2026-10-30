from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd

import torch
from torch import nn


def _group_norm(channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = gcd(channels, min(requested_groups, channels))
    return nn.GroupNorm(groups, channels)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, groups: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels, groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels, groups)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels, groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        output = self.relu(self.norm1(self.conv1(inputs)))
        output = self.norm2(self.conv2(output))
        return self.relu(output + residual)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 4
    num_classes: int = 2
    base_channels: int = 16
    group_norm_groups: int = 8
    dropout: float = 0.30


class CompactResNet3D(nn.Module):
    """A GroupNorm ResNet-18-style encoder sized for a 16 GB T4."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        width = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(config.in_channels, width, kernel_size=5, stride=2, padding=2, bias=False),
            _group_norm(width, config.group_norm_groups),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(width, width, blocks=2, stride=1)
        self.layer2 = self._make_layer(width, width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(width * 2, width * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(width * 4, width * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(width * 8, config.num_classes)
        self.apply(self._initialize)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            BasicBlock3D(in_channels, out_channels, stride, self.config.group_norm_groups)
        ]
        layers.extend(
            BasicBlock3D(out_channels, out_channels, 1, self.config.group_norm_groups)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.pool(output).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(inputs)))

    def export_config(self) -> dict[str, object]:
        return asdict(self.config)
