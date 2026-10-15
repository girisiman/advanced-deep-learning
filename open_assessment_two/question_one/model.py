"""
UNet-style convolutional autoencoder for MNIST (and later CelebA).

Implements encoder–decoder with skip connections.
Same class is used for both clean autoencoding and denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two conv layers with ReLU (and optional BatchNorm)."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_ch))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(-1, nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock + MaxPool. Returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, use_bn)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """Upsample + concat skip + ConvBlock.

    Handles the 7→14 and 14→28 transitions for MNIST.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, use_bn)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # MNIST 7→14 is exact; if sizes ever differ we centre-crop the skip
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[:, :, dh // 2 : dh // 2 + x.shape[-2], dw // 2 : dw // 2 + x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAE(nn.Module):
    """
    Lightweight UNet autoencoder.

    Args:
        in_channels: 1 for MNIST, 3 for CelebA later
        base_ch: starting channel count (default 32 → 32/64/128)
        use_bn: BatchNorm in the conv blocks
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32, use_bn: bool = True):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        # Encoder
        self.down1 = DownBlock(in_channels, c1, use_bn)   # 28 → 14, skip 32
        self.down2 = DownBlock(c1, c2, use_bn)             # 14 → 7,  skip 64
        self.bottleneck = ConvBlock(c2, c3, use_bn)        # 7×7, 128

        # Decoder
        self.up2 = UpBlock(c3, c2, c2, use_bn)             # 7 → 14, cat 64 → 64
        self.up1 = UpBlock(c2, c1, c1, use_bn)             # 14 → 28, cat 32 → 32
        self.final = nn.Conv2d(c1, in_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x, s1 = self.down1(x)          # s1: 28×28 (before pool) or after conv
        x, s2 = self.down2(x)          # s2: 14×14
        x = self.bottleneck(x)         # 7×7

        # Decoder with skips
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.final(x)
        return self.out_act(x)


def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """Add Gaussian noise and clamp to [0, 1]. Used only at training time for DAE."""
    if sigma <= 0:
        return x
    noisy = x + sigma * torch.randn_like(x)
    return noisy.clamp(0.0, 1.0)


if __name__ == "__main__":
    # quick shape check
    model = UNetAE(in_channels=1, base_ch=32)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print(y.shape)  # should be (2, 1, 28, 28)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")
