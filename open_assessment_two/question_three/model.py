import torch
import torch.nn as nn
from needle import NeedleBackbone


class PatchEmbed(nn.Module):
    """Split image into patches and project to d_model.

    For 64×64 image and patch_size=8 → 8×8 = 64 patches.

    Args:
        img_size: spatial size (assumes square)
        patch_size: patch spatial size
        in_channels: 3 for RGB
        d_model: embedding dimension
    """

    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, d_model: int = 128):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, img_size, img_size)
        Returns:
            (batch, n_patches, d_model)
        """
        x = self.proj(x)                          # (B, d_model, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)          # (B, n_patches, d_model)
        return x

class PatchReconHead(nn.Module):
    """Map patch tokens back to an image.

    Args:
        d_model: token dimension
        patch_size: spatial size of each patch
        out_channels: 3 for RGB
        img_size: final image size
    """

    def __init__(self, d_model: int = 128, patch_size: int = 8, out_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches_side = img_size // patch_size
        self.proj = nn.Linear(d_model, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, d_model)
        Returns:
            (batch, out_channels, img_size, img_size)
        """
        B, N, _ = x.shape
        x = self.proj(x)                                      # (B, N, C*p*p)
        x = x.view(B, self.n_patches_side, self.n_patches_side,
                   -1, self.patch_size, self.patch_size)      # (B, H/p, W/p, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()          # (B, C, H/p, p, W/p, p)
        x = x.view(B, -1, self.img_size, self.img_size)       # (B, C, H, W)
        return torch.sigmoid(x)                               # pixels in [0, 1]

class CactusImageAE(nn.Module):
    """Cactus-pure encoder–decoder autoencoder for images.

    PatchEmbed → NeedleBackbone → PatchReconHead
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.backbone = NeedleBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            dropout=dropout,
        )
        self.head = PatchReconHead(d_model, patch_size, in_channels, img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, img_size, img_size)
        Returns:
            reconstructed image, same shape
        """
        tokens = self.patch_embed(x)          # (B, N, d_model)
        # For a pure autoencoder we feed the same tokens as both src and tgt
        features = self.backbone(tokens, tokens)
        return self.head(features)