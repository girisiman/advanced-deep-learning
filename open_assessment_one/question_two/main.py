"""Train the Q2 vanilla Transformer next-word model.

Run from this folder:

    python main.py --epochs 2 --max-batches 20
    python main.py --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Script dir stays first on sys.path so "model" is Q2, not Q1 LSTM.
QUESTION_ONE = Path(__file__).resolve().parent.parent / "question_one"
sys.path.append(str(QUESTION_ONE))

from dataset import build_dataloaders, tokenize
from model import NextWordModel
from utils import evaluate, generate, train_one_epoch

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def parse_args() -> argparse.Namespace:
    """Command-line settings.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Q2 next-word Transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def run_training(
    model: NextWordModel,
    loaders: dict,
    pad_id: int,
    epochs: int,
    lr: float,
    device: torch.device,
    max_batches: int | None,
    grad_clip: float,
) -> dict:
    """Train and evaluate the Transformer.

    Args:
        model: Q2 NextWordModel.
        loaders: Train/val/test loaders from Q1 dataset.py.
        pad_id: <pad> id.
        epochs: Training epochs.
        lr: Adam learning rate.
        device: cpu or cuda.
        max_batches: Optional batch cap.
        grad_clip: Gradient clip norm.

    Returns:
        Test metrics from the best val-PPL checkpoint.
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "transformer_best.pt"
    best_val_ppl = float("inf")

    print("=== train transformer ===")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            pad_id=pad_id,
            device=device,
            max_batches=max_batches,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            pad_id=pad_id,
            device=device,
            max_batches=max_batches,
        )
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_ppl={val_metrics['ppl']:.2f}  "
            f"val_top1={val_metrics['top1']:.3f}"
        )
        if val_metrics["ppl"] < best_val_ppl:
            best_val_ppl = val_metrics["ppl"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        pad_id=pad_id,
        device=device,
        max_batches=max_batches,
    )
    print("test transformer:", test_metrics)
    return test_metrics


def print_table(metrics: dict) -> None:
    """Print the unseen-test row.

    Args:
        metrics: evaluate() dict.
    """
    acc = metrics["acc_at_k"]
    print("\n=== unseen test ===")
    print(f"{'model':<12} {'ppl':>8} {'top1':>8} {'top5':>8} {'k=4':>8} {'k=16':>8}")
    print(
        f"{'transform':<12} {metrics['ppl']:8.2f} {metrics['top1']:8.3f} "
        f"{metrics['top5']:8.3f} {acc.get(4, 0):8.3f} {acc.get(16, 0):8.3f}"
    )


def show_generations(
    model: NextWordModel,
    vocab,
    device: torch.device,
) -> None:
    """Greedy completions from a few seeds.

    Args:
        model: Trained Q2 model.
        vocab: Q1 vocab.
        device: cpu or cuda.
    """
    seeds = [
        "first citizen :",
        "what say you",
        "the king shall",
        "i will not",
    ]
    print("\n=== generate (greedy) ===")
    for seed in seeds:
        tokens = tokenize(text=seed)
        words = generate(
            model=model,
            vocab=vocab,
            seed_tokens=tokens,
            max_new_tokens=12,
            temperature=0.0,
            device=device,
        )
        print("seed:", seed)
        print("gen: ", " ".join(words))
        print()


def main() -> None:
    """Load Q1 data, train the Transformer, report test metrics."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    loaders, vocab, stats = build_dataloaders(
        batch_size=args.batch_size,
        min_freq=args.min_freq,
    )
    print("tokens:", stats["tokens"])
    print("windows:", stats["windows"])
    print("vocab size:", stats["vocab_size"])

    model = NextWordModel(
        vocab_size=stats["vocab_size"],
        embed_size=args.embed_size,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_len=32,
        pad_id=vocab.pad_id(),
        dropout=args.dropout,
    ).to(device)

    test_metrics = run_training(
        model=model,
        loaders=loaders,
        pad_id=vocab.pad_id(),
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        max_batches=args.max_batches,
        grad_clip=args.grad_clip,
    )
    print_table(metrics=test_metrics)
    show_generations(model=model, vocab=vocab, device=device)


if __name__ == "__main__":
    main()
