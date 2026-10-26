"""
Shared Cactus model for image↔text on Flickr30k.

Variants controlled by flags:
  baseline     → use_ffn=False, cross_attn="all"
  ffn          → use_ffn=True,  cross_attn="all"
  cross_last   → use_ffn=False, cross_attn="last"
"""

import torch
import torch.nn as nn
from needle import NeedleBackbone


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, d_model=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class PatchReconHead(nn.Module):
    def __init__(self, d_model=128, patch_size=8, out_channels=3, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.side = img_size // patch_size
        self.proj = nn.Linear(d_model, out_channels * patch_size * patch_size)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        x = x.view(B, self.side, self.side, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return torch.sigmoid(x.view(B, -1, self.img_size, self.img_size))


class CactusMultiModal(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 3,
        img_size: int = 64,
        patch_size: int = 8,
        use_ffn: bool = False,
        cross_attn: str = "all",
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)
        self.recon_head = PatchReconHead(d_model, patch_size, 3, img_size)
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.txt_head = nn.Linear(d_model, vocab_size)

        self.backbone = NeedleBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_encoder_layers=n_layers,
            n_decoder_layers=n_layers,
            use_ffn=use_ffn,
            cross_attn=cross_attn,
        )

    def forward_i2t(self, image, caption_ids):
        mem = self.patch_embed(image)
        tgt = self.tok_embed(caption_ids)
        return self.txt_head(self.backbone(mem, tgt))

    def forward_t2i(self, image, caption_ids):
        mem = self.tok_embed(caption_ids)
        tgt = self.patch_embed(image)
        return self.recon_head(self.backbone(mem, tgt))

    def forward(self, image, caption_ids, mode: str = "i2t"):
        if mode == "i2t":
            return self.forward_i2t(image, caption_ids)
        if mode == "t2i":
            return self.forward_t2i(image, caption_ids)
        raise ValueError(mode)