"""
Cactus Needle-style blocks, built from scratch for Q3.

Pure attention encoder–decoder (no FFN).
Blocks added one by one.

Current contents:
- ZCRMSNorm
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Zero-centred RMSNorm
# ---------------------------------------------------------------------------

class ZCRMSNorm(nn.Module):
    """RMSNorm after subtracting the mean. Scale starts at zero.

    Needle-style: zero-centering + RMS, with a learned scale that is
    initialized to zero so the block starts as a pure normalizer.

    Args:
        dim: last-axis width (d_model or head dim)
        eps: small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # scale starts at 0 → effective multiplier is 1.0 at init
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        x = x - x.mean(dim=-1, keepdim=True)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * (1.0 + self.scale)


import math

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features: (a, b) -> (-b, a).

    Args:
        x: (..., head_dim) with even head_dim
    """
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def rope_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos and sin tables for RoPE.

    Returns:
        cos, sin each shaped (1, 1, seq_len, head_dim)
        so they broadcast over (batch, heads, time, head_dim)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)          # (seq_len, head_dim/2)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat((cos_half, cos_half), dim=-1)      # (seq_len, head_dim)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    cos = cos.to(dtype=dtype or torch.float32)[None, None, :, :]
    sin = sin.to(dtype=dtype or torch.float32)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a query or key tensor.

    Args:
        x:   (batch, heads, time, head_dim)
        cos: broadcastable to x
        sin: broadcastable to x
    """
    return x * cos + rotate_half(x) * sin

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Standard scaled dot-product attention.

    Args:
        query: (batch, heads, q_len, dim)
        key:   (batch, heads, k_len, dim)
        value: (batch, heads, k_len, dim)
        mask:  broadcastable bool tensor, True = do not attend
        dropout: drop probability on attention weights
        training: whether dropout is active

    Returns:
        (batch, heads, q_len, dim)
    """
    dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)

    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)

    if dropout > 0.0 and training:
        weights = torch.nn.functional.dropout(weights, p=dropout)

    return torch.matmul(weights, value)

class GroupedQueryAttention(nn.Module):
    """Multi-head attention with fewer KV heads than query heads (GQA).

    Args:
        d_model: residual width
        n_heads: number of query heads
        n_kv_heads: number of key/value heads (must divide n_heads)
        dropout: attention-weight dropout
        use_rope: apply RoPE to Q and K (True for self-attn, False for cross-attn)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d_model = d_model
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

    def _split_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        # (batch, time, n_heads * dim) -> (batch, n_heads, time, dim)
        batch, time, _ = x.shape
        x = x.view(batch, time, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # Repeat KV heads so they match the number of query heads
        if self.repeats == 1:
            return x
        batch, n_kv, time, dim = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv, self.repeats, time, dim)
        return x.reshape(batch, n_kv * self.repeats, time, dim)

    def forward(
        self,
        query_x: torch.Tensor,
        kv_x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query_x: (batch, q_len, d_model)
            kv_x:    (batch, k_len, d_model)
            mask:    optional bool mask for scaled_dot_product_attention
        Returns:
            (batch, q_len, d_model)
        """
        q = self._split_heads(self.q_proj(query_x), self.n_heads)
        k = self._split_heads(self.k_proj(kv_x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(kv_x), self.n_kv_heads)

        if self.use_rope:
            cos, sin = rope_cos_sin(
                seq_len=q.size(2),
                head_dim=self.head_dim,
                theta=self.rope_theta,
                device=q.device,
                dtype=q.dtype,
            )
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )
        # (batch, heads, q_len, dim) -> (batch, q_len, d_model)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.size(0), out.size(1), -1)
        return self.o_proj(out)
    

class GatedResidual(nn.Module):
    """x + sigmoid(gate) * update

    The gate is a learned parameter initialized to zero.
    At the start of training the residual contribution is small
    (sigmoid(0) = 0.5, but the scale is still learnable from a neutral point).

    Args:
        dim: width of the residual stream (d_model)
    """

    def __init__(self, dim: int):
        super().__init__()
        # scalar gate per channel, starts at 0
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      residual stream (..., dim)
            update: output of the sub-layer (..., dim)
        Returns:
            same shape as x
        """
        return x + torch.sigmoid(self.gate) * update

class EncoderBlock(nn.Module):
    """One pure-attention encoder layer.

    ZCRMSNorm → self-attention (with RoPE) → gated residual
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = ZCRMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=True,          # self-attention uses RoPE
        )
        self.residual = GatedResidual(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
            mask: optional attention mask
        """
        h = self.norm(x)
        update = self.attn(h, h, mask=mask)     # self-attention
        return self.residual(x, update)
    
class DecoderBlock(nn.Module):
    """One pure-attention decoder layer.

    1. ZCRMSNorm → causal self-attention (RoPE) → gated residual
    2. ZCRMSNorm → cross-attention (no RoPE) → gated residual
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        # self-attention
        self.norm1 = ZCRMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=True,
        )
        self.residual1 = GatedResidual(d_model)

        # cross-attention
        self.norm2 = ZCRMSNorm(d_model)
        self.cross_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            use_rope=False,         # cross-attention: no RoPE
        )
        self.residual2 = GatedResidual(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (batch, tgt_len, d_model)
            memory: (batch, src_len, d_model)  — encoder output
        """
        # causal self-attention
        h = self.norm1(x)
        update = self.self_attn(h, h, mask=self_mask)
        x = self.residual1(x, update)

        # cross-attention to encoder memory
        h = self.norm2(x)
        update = self.cross_attn(h, memory, mask=cross_mask)
        x = self.residual2(x, update)
        return x

class NeedleBackbone(nn.Module):
    """Stack of pure-attention encoder and decoder blocks.

    Args:
        d_model: residual width
        n_heads: query heads
        n_kv_heads: key/value heads
        n_encoder_layers: number of EncoderBlocks
        n_decoder_layers: number of DecoderBlocks
        dropout: attention dropout
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, n_kv_heads, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, n_kv_heads, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.final_norm = ZCRMSNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len, d_model)  — encoder input (patch embeddings)
            tgt: (batch, tgt_len, d_model)  — decoder input
        Returns:
            (batch, tgt_len, d_model)
        """
        # Encoder
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, mask=src_mask)

        # Decoder
        x = tgt
        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask=tgt_mask, cross_mask=cross_mask)

        return self.final_norm(x)