from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import build_dataloaders, tokenize
from model import NextWordModel
from utils import evaluate, generate, train_one_epoch

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

def parse_args() -> argparse.Namespace:
    """Command-line settings for a CPU-friendly run.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Q1 next-word LSTM / RNN")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch (smoke test).",
    )
    parser.add_argument(
        "--skip-rnn",
        action="store_true",
        help="Train only LSTM (faster).",
    )
    return parser.parse_args()

def make_model(
    vocab_size: int,
    pad_id: int,
    embed_size: int,
    hidden_size: int,
    dropout: float,
    backbone: str,
    device: torch.device,
) -> NextWordModel:
    """Build one next-word model.

    Args:
        vocab_size: Size of the training vocab.
        pad_id: <pad> id.
        embed_size: Embedding width.
        hidden_size: RNN/LSTM width.
        dropout: Dropout before the vocab head.
        backbone: "lstm" or "rnn".
        device: cpu or cuda.

    Returns:
        Model on the given device.
    """
    model = NextWordModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        backbone=backbone,
        pad_id=pad_id,
        dropout=dropout,
    )
    return model.to(device)

def run_training(
    model: NextWordModel,
    loaders: dict,
    pad_id: int,
    epochs: int,
    lr: float,
    device: torch.device,
    max_batches: int | None,
    grad_clip: float,
    name: str,
) -> dict:
    """Train for several epochs and keep the best val-PPL checkpoint.

    Args:
        model: NextWordModel.
        loaders: Dict with train/val/test.
        pad_id: Padding id.
        epochs: Number of passes over train.
        lr: Adam learning rate.
        device: cpu or cuda.
        max_batches: Optional batch cap per epoch.
        grad_clip: Gradient clip norm.
        name: "lstm" or "rnn", used in the checkpoint filename.

    Returns:
        Test metrics from the best val epoch.
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{name}_best.pt"
    best_val_ppl = float("inf")

    print(f"\n=== train {name} ===")
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
    print(f"test {name}: {test_metrics}")
    return test_metrics

def print_table(results: dict) -> None:
    """Print the exam-style comparison table.

    Args:
        results: Mapping backbone name -> evaluate() dict.
    """
    print("\n=== unseen test ===")
    print(f"{'model':<8} {'ppl':>8} {'top1':>8} {'top5':>8} {'k=4':>8} {'k=16':>8}")
    for name, metrics in results.items():
        acc = metrics["acc_at_k"]
        print(
            f"{name:<8} {metrics['ppl']:8.2f} {metrics['top1']:8.3f} "
            f"{metrics['top5']:8.3f} {acc.get(4, 0):8.3f} {acc.get(16, 0):8.3f}"
        )


def show_generations(
    model: NextWordModel,
    vocab,
    device: torch.device,
) -> None:
    """Print greedy completions from a few seeds.

    Args:
        model: Trained model.
        vocab: Dataset vocab.
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
    """Load data, train LSTM (and optional RNN), report test metrics."""
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

    results = {}
    backbones = ["lstm"] if args.skip_rnn else ["lstm", "rnn"]
    last_lstm = None

    for backbone in backbones:
        model = make_model(
            vocab_size=stats["vocab_size"],
            pad_id=vocab.pad_id(),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            backbone=backbone,
            device=device,
        )
        test_metrics = run_training(
            model=model,
            loaders=loaders,
            pad_id=vocab.pad_id(),
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            max_batches=args.max_batches,
            grad_clip=args.grad_clip,
            name=backbone,
        )
        results[backbone] = test_metrics
        if backbone == "lstm":
            last_lstm = model

    print_table(results=results)
    if last_lstm is not None:
        show_generations(model=last_lstm, vocab=vocab, device=device)


if __name__ == "__main__":
    main()



