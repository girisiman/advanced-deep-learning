from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

QUESTION_TWO = Path(__file__).resolve().parent.parent / "question_two"
TRANSFORMER_FILE = QUESTION_TWO / "transformer.py"

spec = importlib.util.spec_from_file_location("q2_transformer", TRANSFORMER_FILE)
q2_transformer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q2_transformer)
TransformerBackbone = q2_transformer.TransformerBackbone


class NextWordModel(nn.Module):
    """Promptable LM: embedding + Q2 Transformer + vocab head."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        max_len: int = 48,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        """Build the Q3 model.

        Args:
            vocab_size: MixVocab size (English + Nepali + prefixes).
            embed_size: d_model.
            n_heads: Attention heads.
            d_ff: FFN width.
            n_layers: Transformer blocks.
            max_len: Must cover prefix + sentence + answer.
            pad_id: <pad> id.
            dropout: Dropout inside the backbone.
        """
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_id,
        )
        self.backbone = TransformerBackbone(
            d_model=embed_size,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-token logits for a prompted sequence.

        Args:
            input_ids: Shape (batch_size, time_steps).

        Returns:
            logits: Shape (batch_size, time_steps, vocab_size).
        """
        x = self.embedding(input_ids)
        output, extra = self.backbone.forward(x=x, hidden=None)
        logits = self.lm_head(output)
        return logits