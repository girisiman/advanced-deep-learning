"""Needle-style Simple Attention Network for Question 4.

Encoder-decoder built from scratch. No nn.Transformer / nn.MultiheadAttention.

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZCRMSNorm(nn.Module):
    """RMSNorm with a scale that starts at zero.

    Vaswani uses LayerNorm: subtract mean, divide by std, then affine.
    Needle uses RMS only, and stores the scale as (1 + gamma) so a zero
    init still means "unit RMS", not "output is all zeros".

    Args:
        dim: Last-axis width (d_model or head dim).
        eps: Floor under the RMS.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over the last dimension.

        Args:
            x: (..., dim)

        Returns:
            Same shape as x.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * (1.0 + self.scale)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features: (a, b) -> (-b, a).

    Args:
        x: (..., head_dim) with even head_dim.

    Returns:
        Same shape as x.
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
    """Cos/sin tables for RoPE.

    Args:
        seq_len: Number of positions.
        head_dim: Per-head width. Must be even.
        theta: Base wavelength (Llama / Qwen default 10000).
        device: Where to put the tables.
        dtype: Activation dtype.

    Returns:
        cos, sin each (1, 1, seq_len, head_dim).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    cos = cos.to(dtype=dtype or torch.float32)[None, None, :, :]
    sin = sin.to(dtype=dtype or torch.float32)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a query or key.

    Args:
        x: (batch, heads, time, head_dim)
        cos: broadcastable to x
        sin: broadcastable to x

    Returns:
        Rotated x, same shape.
    """
    return x * cos + rotate_half(x) * sin

def make_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """True where the token is pad. Shape (batch, time)."""
    return token_ids.eq(pad_id)

def make_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """True above the diagonal. Position t may see 0..t only."""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )

def merge_masks(pad_mask, causal_mask, n_heads: int):
    """(batch, heads, query_len, key_len). True = block."""
    mask = None
    if pad_mask is not None:
        mask = pad_mask[:, None, None, :]
    if causal_mask is not None:
        causal = causal_mask[None, None, :, :]
        mask = causal if mask is None else (mask | causal)
    if mask is None:
        return None
    return mask.expand(mask.size(0), n_heads, mask.size(2), mask.size(3))

def scaled_dot_product_attention(query, key, value, mask=None, dropout=0.0, training=False):
    """softmax(QK^T / sqrt(d)) V. mask True = do not attend."""
    dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout > 0.0 and training:
        weights = F.dropout(weights, p=dropout)
    return torch.matmul(weights, value)

class GroupedQueryAttention(nn.Module):
    """Multi-head attention with fewer KV heads than query heads.

    n_heads must divide d_model. n_heads must divide by n_kv_heads.
    Each KV head is repeated so it lines up with the query heads.

    Self-attention: pass the same tensor as query_x and kv_x.
    Cross-attention: query_x is the decoder, kv_x is encoder memory.

    Args:
        d_model: Residual width.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        dropout: Dropout on attention weights.
        use_rope: Apply RoPE to Q and K. Off for cross-attention.
        rope_theta: RoPE base wavelength.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must divide by n_kv_heads")
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
        """Split the last axis into heads.

        Args:
            x: (batch, time, n_heads * head_dim)
            n_heads: How many heads that axis holds.

        Returns:
            (batch, n_heads, time, head_dim)
        """
        batch, time, _ = x.shape
        x = x.view(batch, time, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads so there is one per query head.

        Args:
            x: (batch, n_kv_heads, time, head_dim)

        Returns:
            (batch, n_heads, time, head_dim)
        """
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
        """Attend query_x to kv_x.

        Args:
            query_x: (batch, q_len, d_model)
            kv_x: (batch, k_len, d_model)
            mask: (batch, heads, q_len, k_len), True = block. Optional.

        Returns:
            (batch, q_len, d_model)
        """
        q = self._split_heads(self.q_proj(query_x), self.n_heads)
        k = self._split_heads(self.k_proj(kv_x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(kv_x), self.n_kv_heads)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if self.use_rope:
            q_cos, q_sin = rope_cos_sin(
                q.size(2), self.head_dim, self.rope_theta, q.device, q.dtype
            )
            k_cos, k_sin = rope_cos_sin(
                k.size(2), self.head_dim, self.rope_theta, k.device, k.dtype
            )
            q = apply_rope(q, q_cos, q_sin)
            k = apply_rope(k, k_cos, k_sin)
        attended = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, training=self.training
        )
        batch, _, q_len, _ = attended.shape
        attended = attended.permute(0, 2, 1, 3).contiguous()
        attended = attended.view(batch, q_len, self.n_heads * self.head_dim)
        return self.o_proj(attended)

class GatedResidual(nn.Module):
    """Mix the residual stream with a new update.

    General practice: x + update.
    Needle: x + sigmoid(g) * update, with g starting at 0.
    sigmoid(0) is 0.5, so a new block writes at half volume at step 0.

    Returns:
        Same shape as residual.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Write update into residual.

        Args:
            residual: Incoming stream x.
            update: Attention (or FFN) output.

        Returns:
            residual + sigmoid(gate) * update
        """
        return residual + torch.sigmoid(self.gate) * update

class FeedForward(nn.Module):
    """Two-layer MLP applied at each position.

    General practice (Gemma/Qwen): every block has one of these after
    attention, often SwiGLU. Needle SAN deletes it. We keep the class so
    use_ffn=True can turn the classic block back on.

    Args:
        d_model: Residual width.
        d_ff: Hidden width.
        dropout: Dropout after the nonlinearity.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pointwise MLP.

        Args:
            x: (batch, time, d_model)

        Returns:
            Same shape as x.
        """
        return self.fc2(self.dropout(torch.nn.functional.gelu(self.fc1(x))))

class EncoderLayer(nn.Module):
    """One source block: bidirectional self-attention, optional FFN.

    Needle path (use_ffn=False):
        Norm -> GQA + RoPE -> gated residual
    Classic path (use_ffn=True):
        the same, then Norm -> FFN -> gated residual

    Args:
        d_model: Residual width.
        n_heads: Query heads.
        n_kv_heads: KV heads.
        d_ff: FFN hidden width, used only if use_ffn is True.
        dropout: Attention / FFN dropout.
        use_ffn: False = Needle SAN. True = classic block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int = 512,
        dropout: float = 0.0,
        use_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.use_ffn = use_ffn
        self.norm_attn = ZCRMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout=dropout, use_rope=True
        )
        self.gate_attn = GatedResidual()
        if use_ffn:
            self.norm_ffn = ZCRMSNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
            self.gate_ffn = GatedResidual()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Encode one layer.

        Args:
            x: (batch, src_len, d_model)
            mask: Self-attention mask, or None.

        Returns:
            (batch, src_len, d_model)
        """
        nx = self.norm_attn(x)
        x = self.gate_attn(x, self.self_attn(nx, nx, mask=mask))
        if self.use_ffn:
            x = self.gate_ffn(x, self.ffn(self.norm_ffn(x)))
        return x

class DecoderLayer(nn.Module):
    """One target block.

    Always has causal self-attention. Cross-attention is optional so the
    exam can pass encoder states into one layer or into every layer.

    Args:
        d_model: Residual width.
        n_heads: Query heads.
        n_kv_heads: KV heads.
        d_ff: FFN width if use_ffn is True.
        dropout: Dropout.
        use_ffn: Classic FFN on/off.
        use_cross_attn: Whether this layer reads the encoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int = 512,
        dropout: float = 0.0,
        use_ffn: bool = False,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_ffn = use_ffn

        self.norm_self = ZCRMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout=dropout, use_rope=True
        )
        self.gate_self = GatedResidual()

        if use_cross_attn:
            self.norm_cross_q = ZCRMSNorm(d_model)
            self.norm_cross_kv = ZCRMSNorm(d_model)
            self.cross_attn = GroupedQueryAttention(
                d_model, n_heads, n_kv_heads, dropout=dropout, use_rope=False
            )
            self.gate_cross = GatedResidual()

        if use_ffn:
            self.norm_ffn = ZCRMSNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
            self.gate_ffn = GatedResidual()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode one layer.

        Args:
            x: Target states (batch, tgt_len, d_model).
            memory: Encoder output (batch, src_len, d_model).
            self_mask: Causal (+ pad) mask for the target.
            cross_mask: Source-pad mask for cross-attention.

        Returns:
            (batch, tgt_len, d_model)
        """
        nx = self.norm_self(x)
        x = self.gate_self(x, self.self_attn(nx, nx, mask=self_mask))
        if self.use_cross_attn:
            q = self.norm_cross_q(x)
            kv = self.norm_cross_kv(memory)
            x = self.gate_cross(x, self.cross_attn(q, kv, mask=cross_mask))
        if self.use_ffn:
            x = self.gate_ffn(x, self.ffn(self.norm_ffn(x)))
        return x

class NeedleEncoder(nn.Module):
    """Stack of encoder layers plus a final norm.

    Args:
        n_layers: Encoder depth.
        d_model: Width.
        n_heads: Query heads.
        n_kv_heads: KV heads.
        d_ff: FFN width.
        dropout: Dropout.
        use_ffn: Classic FFN on/off.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int = 512,
        dropout: float = 0.0,
        use_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout, use_ffn)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = ZCRMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Encode the source.

        Args:
            x: (batch, src_len, d_model)
            mask: Self-attention mask.

        Returns:
            memory (batch, src_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x)


class NeedleDecoder(nn.Module):
    """Stack of decoder layers.

    cross_attn='all' — every layer reads the encoder.
    cross_attn='one' — only the last layer reads the encoder.

    Args:
        n_layers: Decoder depth.
        d_model: Width.
        n_heads: Query heads.
        n_kv_heads: KV heads.
        d_ff: FFN width.
        dropout: Dropout.
        use_ffn: Classic FFN on/off.
        cross_attn: 'all' or 'one'.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int = 512,
        dropout: float = 0.0,
        use_ffn: bool = False,
        cross_attn: str = "all",
    ) -> None:
        super().__init__()
        if cross_attn not in {"all", "one"}:
            raise ValueError("cross_attn must be 'all' or 'one'")
        self.cross_attn = cross_attn
        layers = []
        for i in range(n_layers):
            use_cross = (cross_attn == "all") or (i == n_layers - 1)
            layers.append(
                DecoderLayer(
                    d_model, n_heads, n_kv_heads, d_ff, dropout, use_ffn,
                    use_cross_attn=use_cross,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = ZCRMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None,
        cross_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode the target against encoder memory.

        Args:
            x: (batch, tgt_len, d_model)
            memory: Encoder output.
            self_mask: Causal target mask.
            cross_mask: Source pad mask.

        Returns:
            (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.final_norm(x)


class NeedleBackbone(nn.Module):
    """Encoder + decoder used by TranslationModel.

    Args:
        n_enc: Encoder layers.
        n_dec: Decoder layers.
        d_model: Width.
        n_heads: Query heads.
        n_kv_heads: KV heads.
        d_ff: FFN width.
        dropout: Dropout.
        use_ffn: Classic FFN on/off.
        cross_attn: 'all' or 'one'.
    """

    def __init__(
        self,
        n_enc: int = 3,
        n_dec: int = 3,
        d_model: int = 256,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_ffn: bool = False,
        cross_attn: str = "all",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder = NeedleEncoder(
            n_enc, d_model, n_heads, n_kv_heads, d_ff, dropout, use_ffn
        )
        self.decoder = NeedleDecoder(
            n_dec, d_model, n_heads, n_kv_heads, d_ff, dropout, use_ffn, cross_attn
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        src_pad_mask: torch.Tensor | None,
        tgt_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run encoder then decoder.

        Args:
            src_emb: (batch, src_len, d_model)
            tgt_emb: (batch, tgt_len, d_model)
            src_pad_mask: (batch, src_len) True = pad.
            tgt_pad_mask: (batch, tgt_len) True = pad.

        Returns:
            Decoder states (batch, tgt_len, d_model).
        """
        tgt_len = tgt_emb.size(1)
        enc_mask = merge_masks(src_pad_mask, None, self.n_heads)
        causal = make_causal_mask(tgt_len, device=src_emb.device)
        dec_self_mask = merge_masks(tgt_pad_mask, causal, self.n_heads)
        cross_mask = merge_masks(src_pad_mask, None, self.n_heads)
        memory = self.encoder(src_emb, enc_mask)
        return self.decoder(tgt_emb, memory, dec_self_mask, cross_mask)

