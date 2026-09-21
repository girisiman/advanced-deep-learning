import torch
import torch.nn as nn

from transformer import TransformerBackbone


class NextWordModel(nn.Module):
    """Embedding + vanilla Transformer + vocab head."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        max_len: int = 32,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        """Build the Q2 language model.

        Args:
            vocab_size: Number of tokens.
            embed_size: Word-vector width (also d_model).
            n_heads: Attention heads per block.
            d_ff: FFN hidden width.
            n_layers: Number of Transformer blocks.
            max_len: Longest window the backbone allows.
            pad_id: <pad> id; embedding row stays zero.
            dropout: Dropout inside the backbone.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
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
        self.lm_head = nn.Linear(
            in_features=embed_size,
            out_features=vocab_size,
            bias=True,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Predict next-word scores at every position.

        Args:
            input_ids: Token ids, shape (batch_size, time_steps).

        Returns:
            logits: Shape (batch_size, time_steps, vocab_size).
        """
        x = self.embedding(input_ids)
        output, extra = self.backbone.forward(x=x, hidden=None)
        logits = self.lm_head(output)
        return logits