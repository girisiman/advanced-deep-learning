"""Train the Q3 promptable Transformer.

    python main.py --epochs 1 --max-batches 20
    python main.py --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
QUESTION_ONE = HERE.parent / "question_one"
sys.path.insert(0, str(HERE))
sys.path.append(str(QUESTION_ONE))

from mixed_data import BITEXT_PAIRS, build_dataloaders
from model import NextWordModel
from prompts import format_fill, format_next, format_translate
from utils import evaluate, generate, train_one_epoch

CHECKPOINT_DIR = HERE / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q3 promptable Transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def show_task_examples(model, vocab, device) -> None:
    print("\n=== prompted generations ===")

    next_seed = format_next(tokens=["the", "king", "shall"])
    print("NEXT seed:", next_seed)
    print("NEXT gen: ", generate(model, vocab, next_seed, 6, 0.0, device))

    fill_full = format_fill(
        tokens=["the", "king", "shall", "come"],
        mask_index=2,
    )
    fill_prompt = fill_full[:-1]
    print("FILL seed:", fill_prompt)
    print("FILL gen: ", generate(model, vocab, fill_prompt, 3, 0.0, device))

    en, ne = BITEXT_PAIRS[1]
    en2ne = format_translate(en.split(), ne.split(), "en2ne")
    en2ne_prompt = en2ne[: en2ne.index("[sep]") + 1]
    print("EN2NE seed:", en2ne_prompt)
    print("EN2NE gen: ", generate(model, vocab, en2ne_prompt, 6, 0.0, device))

    ne2en = format_translate(ne.split(), en.split(), "ne2en")
    ne2en_prompt = ne2en[: ne2en.index("[sep]") + 1]
    print("NE2EN seed:", ne2en_prompt)
    print("NE2EN gen: ", generate(model, vocab, ne2en_prompt, 6, 0.0, device))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    loaders, vocab, stats = build_dataloaders(batch_size=args.batch_size)
    print("stats:", stats)

    model = NextWordModel(
        vocab_size=stats["vocab_size"],
        embed_size=args.embed_size,
        max_len=48,
        pad_id=vocab.pad_id(),
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "promptable_best.pt"
    best_ppl = float("inf")

    print("=== train promptable transformer ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            pad_id=vocab.pad_id(),
            device=device,
            max_batches=args.max_batches,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            pad_id=vocab.pad_id(),
            device=device,
            max_batches=args.max_batches,
        )
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_ppl={val_metrics['ppl']:.2f}"
        )
        if val_metrics["ppl"] < best_ppl:
            best_ppl = val_metrics["ppl"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        pad_id=vocab.pad_id(),
        device=device,
        max_batches=args.max_batches,
    )
    print("test:", test_metrics)
    show_task_examples(model=model, vocab=vocab, device=device)


if __name__ == "__main__":
    main()