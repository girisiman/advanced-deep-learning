import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_causal_mask(time_steps: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask of -inf so position t cannot see t+1, t+2, ...

    Args:
        time_steps: Sequence length T.
        device: Device of Q/K/V.

    Returns:
        Mask of shape (1, 1, T, T), ready to add to attention scores.
    """
    # True above the diagonal = future tokens
    future = torch.triu(
        torch.ones(time_steps, time_steps, device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask = torch.zeros(time_steps, time_steps, device=device)
    mask = mask.masked_fill(future, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-head-style attention. Works on any leading batch dims.

    Args:
        query: Shape (..., T_q, d).
        key: Shape (..., T_k, d).
        value: Shape (..., T_k, d_v).
        mask: Optional additive mask, broadcastable to (..., T_q, T_k).
              Use 0 for keep, -inf for block.

    Returns:
        output: Shape (..., T_q, d_v).
        weights: Attention weights, shape (..., T_q, T_k).
    """
    d = query.size(dim=-1)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d)
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, value)
    return output, weights

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """Create Q/K/V and output projections.

        Args:
            d_model: Embedding width. Must be divisible by n_heads.
            n_heads: Number of attention heads.
            dropout: Dropout on attention output.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, T, d_model) -> (batch, n_heads, T, d_head)."""
        batch, time_steps, d_model = x.shape
        x = x.view(batch, time_steps, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_heads, T, d_head) -> (batch, T, d_model)."""
        batch, n_heads, time_steps, d_head = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, time_steps, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the sequence with a causal mask.

        Args:
            x: Shape (batch_size, time_steps, d_model).

        Returns:
            Tensor of the same shape as x.
        """
        batch, time_steps, d_model = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        mask = make_causal_mask(time_steps=time_steps, device=x.device)
        attn_out, weights = scaled_dot_product_attention(
            query=q, key=k, value=v, mask=mask
        )
        merged = self._merge_heads(attn_out)
        output = self.out_proj(merged)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    """Position-wise two-layer MLP."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """Create the two linear maps.

        Args:
            d_model: Embedding width.
            d_ff: Hidden width inside the FFN.
            dropout: Dropout on the FFN output.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN at every time step independently.

        Args:
            x: Shape (batch_size, time_steps, d_model).

        Returns:
            Tensor of the same shape as x.
        """
        hidden = F.relu(self.fc1(x))
        output = self.fc2(hidden)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    """One causal Transformer layer: attention + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """Build attention, FFN, and two layer norms.

        Args:
            d_model: Embedding width.
            n_heads: Attention heads.
            d_ff: FFN hidden width.
            dropout: Dropout inside attention and FFN.
        """
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one block.

        Args:
            x: Shape (batch_size, time_steps, d_model).

        Returns:
            Tensor of the same shape as x.
        """
        attn_out = self.self_attn.forward(x=self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn.forward(x=self.norm2(x))
        x = x + ffn_out
        return x

class TransformerBackbone(nn.Module):
    """Causal Transformer stack used as a language-model backbone."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_len: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Create position embeddings and N Transformer blocks.

        Args:
            d_model: Embedding width.
            n_heads: Heads per block.
            d_ff: FFN hidden width.
            n_layers: Number of blocks.
            max_len: Longest sequence this backbone accepts.
            dropout: Dropout inside blocks.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_embed = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        hidden=None,
    ) -> tuple[torch.Tensor, None]:
        """Run the stack.

        Args:
            x: Token embeddings, shape (batch_size, time_steps, d_model).
            hidden: Ignored. Present so the call matches RNN/LSTM.

        Returns:
            output: Shape (batch_size, time_steps, d_model).
            extra: None.
        """
        batch, time_steps, d_model = x.shape
        if time_steps > self.max_len:
            raise ValueError("sequence longer than max_len")

        positions = torch.arange(time_steps, device=x.device)
        x = x + self.pos_embed(positions)

        for layer in self.layers:
            x = layer.forward(x=x)

        output = self.final_norm(x)
        return output, None