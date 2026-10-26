"""
Cactus Needle-style blocks for Q3/Q4.

Pure attention by default; optional FFN and cross-attn control for the Q4 experiment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ZCRMSNorm
# ---------------------------------------------------------------------------

class ZCRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * (1.0 + self.scale)


# ---------------------------------------------------------------------------
# 2. RoPE
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rope_cos_sin(seq_len: int, head_dim: int, theta: float = 10000.0,
                 device=None, dtype=None):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    cos = cos.to(dtype=dtype or torch.float32)[None, None, :, :]
    sin = sin.to(dtype=dtype or torch.float32)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# 3. Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(query, key, value, mask=None, dropout=0.0, training=False):
    dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout > 0.0 and training:
        weights = F.dropout(weights, p=dropout)
    return torch.matmul(weights, value)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0, use_rope=True, rope_theta=10000.0):
        super().__init__()
        if d_model % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError("invalid head configuration")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.repeats = n_heads // n_kv_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def _split_heads(self, x, n_heads):
        B, T, _ = x.shape
        x = x.view(B, T, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _repeat_kv(self, x):
        if self.repeats == 1:
            return x
        B, n_kv, T, D = x.shape
        x = x[:, :, None, :, :].expand(B, n_kv, self.repeats, T, D)
        return x.reshape(B, n_kv * self.repeats, T, D)

    def forward(self, query_x, kv_x, mask=None):
        q = self._split_heads(self.q_proj(query_x), self.n_heads)
        k = self._split_heads(self.k_proj(kv_x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(kv_x), self.n_kv_heads)

        if self.use_rope:
            cos, sin = rope_cos_sin(q.size(2), self.head_dim, self.rope_theta, q.device, q.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        out = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout, training=self.training)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.size(0), out.size(2), -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 4. Gated residual + optional FFN
# ---------------------------------------------------------------------------

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x, update):
        return x + torch.sigmoid(self.gate) * update


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Linear(d_model * mult, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 5. Blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0, use_ffn=False):
        super().__init__()
        self.use_ffn = use_ffn
        self.norm = ZCRMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.res = GatedResidual(d_model)
        if use_ffn:
            self.norm_ff = ZCRMSNorm(d_model)
            self.ffn = FeedForward(d_model)
            self.res_ff = GatedResidual(d_model)

    def forward(self, x, mask=None):
        h = self.norm(x)
        x = self.res(x, self.attn(h, h, mask=mask))
        if self.use_ffn:
            h = self.norm_ff(x)
            x = self.res_ff(x, self.ffn(h))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0, use_ffn=False, do_cross=True):
        super().__init__()
        self.do_cross = do_cross
        self.use_ffn = use_ffn

        self.norm1 = ZCRMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=True)
        self.res1 = GatedResidual(d_model)

        if do_cross:
            self.norm2 = ZCRMSNorm(d_model)
            self.cross_attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, use_rope=False)
            self.res2 = GatedResidual(d_model)

        if use_ffn:
            self.norm_ff = ZCRMSNorm(d_model)
            self.ffn = FeedForward(d_model)
            self.res_ff = GatedResidual(d_model)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        h = self.norm1(x)
        x = self.res1(x, self.self_attn(h, h, mask=self_mask))

        if self.do_cross:
            h = self.norm2(x)
            x = self.res2(x, self.cross_attn(h, memory, mask=cross_mask))

        if self.use_ffn:
            h = self.norm_ff(x)
            x = self.res_ff(x, self.ffn(h))
        return x


# ---------------------------------------------------------------------------
# 6. Backbone
# ---------------------------------------------------------------------------

class NeedleBackbone(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=8,
        n_kv_heads=2,
        n_encoder_layers=3,
        n_decoder_layers=3,
        dropout=0.0,
        use_ffn=False,
        cross_attn="all",
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, n_kv_heads, dropout, use_ffn=use_ffn)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(
                d_model, n_heads, n_kv_heads, dropout,
                use_ffn=use_ffn,
                do_cross=(cross_attn == "all" or i == n_decoder_layers - 1),
            )
            for i in range(n_decoder_layers)
        ])
        self.final_norm = ZCRMSNorm(d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, mask=src_mask)
        x = tgt
        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask=tgt_mask, cross_mask=cross_mask)
        return self.final_norm(x)