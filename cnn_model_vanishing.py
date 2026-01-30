import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Optional


def init_weights_xavier(module: nn.Module) -> None:
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)


def init_weights_he(module: nn.Module) -> None:
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            init.zeros_(module.bias)


def init_weights_small(module: nn.Module) -> None:
    """Very small initialization to promote vanishing gradients"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.normal_(module.weight, mean=0.0, std=0.1)
        if module.bias is not None:
            init.zeros_(module.bias)


class VanishingCNN(nn.Module):
    """
    Deep CNN with sigmoid activations intended to exhibit vanishing gradients.
    """

    def __init__(
        self,
        num_conv_layers: int = 15,
        base_channels: int = 16,
        num_classes: int = 10,
        init_method: str = "small",
    ) -> None:
        super().__init__()

        self.num_conv_layers = num_conv_layers
        self.base_channels = base_channels

        conv_layers = []
        in_channels = 3
        for i in range(num_conv_layers):
            out_channels = base_channels * (2 ** min(i // 3, 3))
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            conv_layers.append(conv)
            in_channels = out_channels

        self.conv_layers = nn.ModuleList(conv_layers)

        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pooling operations: 32 -> 4 (pool after layers 2, 5, 8)
        final_spatial = 8
        self.fc1 = nn.Linear(in_channels * final_spatial * final_spatial, 256)
        self.fc2 = nn.Linear(256, num_classes)

        if init_method == "xavier":
            self.apply(init_weights_xavier)
        elif init_method == "he":
            self.apply(init_weights_he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = torch.sigmoid(x)
            if idx in (6, 13):
                x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Used for mitigation of vanishing gradients
class ResidualCNN(nn.Module):
    """
    CNN with residual connections and ReLU activations to mitigate vanishing gradients.
    """

    def __init__(
        self,
        num_blocks: int = 6,
        base_channels: int = 32,
        num_classes: int = 10,
        init_method: str = "he",
    ) -> None:
        super().__init__()

        self.stem = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        blocks = []
        in_channels = base_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** (i // 2))
            downsample = None
            if in_channels != out_channels:
                downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            block = ResidualBlock(in_channels, out_channels, downsample=downsample)
            blocks.append(block)
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

        if init_method == "xavier":
            self.apply(init_weights_xavier)
        elif init_method == "he":
            self.apply(init_weights_he)
        elif init_method == "small":
            self.apply(init_weights_small)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = F.relu(out + identity)
        return out
