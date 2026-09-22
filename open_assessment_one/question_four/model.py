"""Translation wrapper around the Needle encoder-decoder.

One embedding table is shared by source, target, and the vocab head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from needle import NeedleBackbone, make_padding_mask


class TranslationModel(nn.Module):
    """Encoder-decoder next-token model for EN↔NE.

    Args:
        vocab_size: Shared tokenizer vocab size.
        d_model: Residual width.
        n_heads: Query heads.
        n_kv_heads: KV heads.
        n_enc: Encoder layers.
        n_dec: Decoder layers.
        d_ff: FFN width if use_ffn is True.
        dropout: Dropout.
        pad_id: Tokenizer pad id.
        bos_id: Decoder start id.
        eos_id: Decoder stop id.
        use_ffn: False = Needle SAN. True = classic block.
        cross_attn: 'all' or 'one'.
        tie_embeddings: Reuse embed.weight as the output head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        n_enc: int = 3,
        n_dec: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        use_ffn: bool = False,
        cross_attn: str = "all",
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model
        self.cross_attn = cross_attn
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_dropout = nn.Dropout(dropout)
        self.backbone = NeedleBackbone(
            n_enc=n_enc,
            n_dec=n_dec,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_ffn=use_ffn,
            cross_attn=cross_attn,
        )
        self.vocab_head = None if tie_embeddings else nn.Linear(d_model, vocab_size, bias=False)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up tokens and scale by sqrt(d_model).

        Args:
            token_ids: (batch, time)

        Returns:
            (batch, time, d_model)
        """
        return self.embed_dropout(self.embed(token_ids) * (self.d_model ** 0.5))

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project decoder states onto the shared vocab.

        Args:
            hidden: (batch, tgt_len, d_model)

        Returns:
            (batch, tgt_len, vocab_size)
        """
        if self.vocab_head is None:
            return F.linear(hidden, self.embed.weight)
        return self.vocab_head(hidden)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass.

        tgt_ids is decoder input (usually BOS + target[:-1]).
        Labels are shifted by the caller.

        Args:
            src_ids: (batch, src_len)
            tgt_ids: (batch, tgt_len)

        Returns:
            logits (batch, tgt_len, vocab_size)
        """
        hidden = self.backbone(
            self._embed(src_ids),
            self._embed(tgt_ids),
            make_padding_mask(src_ids, self.pad_id),
            make_padding_mask(tgt_ids, self.pad_id),
        )
        return self._logits(hidden)

    @torch.no_grad()
    def generate(self, src_ids: torch.Tensor, max_new_tokens: int = 64) -> torch.Tensor:
        """Greedy decode one batch.

        Args:
            src_ids: (batch, src_len)
            max_new_tokens: Stop after this many tokens or at EOS.

        Returns:
            Generated ids including BOS.
        """
        self.eval()
        batch = src_ids.size(0)
        device = src_ids.device
        ys = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            next_id = self.forward(src_ids, ys)[:, -1, :].argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, self.pad_id), next_id)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
            finished = finished | next_id.eq(self.eos_id)
            if bool(finished.all()):
                break
        return ys


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count.

    Args:
        model: Any module.

    Returns:
        Number of parameters that require grad.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)