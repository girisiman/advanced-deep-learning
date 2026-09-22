"""Train the Q4 Needle translator.

    python -u main.py --epochs 1 --max-batches 20
    python -u main.py --pairs opus --max-pairs 32 --epochs 1 --max-batches 2
    python -u main.py --pairs opus --epochs 8 --cross-attn all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data import SMOKE_PAIRS, build_dataloaders, format_pair, load_opus_sample, load_tokenizer
from model import TranslationModel, count_parameters

HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = HERE / "checkpoints"


def parse_args():
    """Command-line settings."""
    parser = argparse.ArgumentParser(description="Q4 Needle EN-NE train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-enc", type=int, default=2)
    parser.add_argument("--n-dec", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--cross-attn", choices=("all", "one"), default="all")
    parser.add_argument("--use-ffn", action="store_true")
    parser.add_argument("--pairs", choices=("smoke", "opus"), default="smoke")
    parser.add_argument("--max-pairs", type=int, default=8000)
    return parser.parse_args()


def batch_loss(model, batch, pad_id, device):
    """Mean token NLL, pad ignored."""
    src = batch["src_ids"].to(device)
    tgt_in = batch["tgt_in"].to(device)
    labels = batch["labels"].to(device)
    logits = model(src, tgt_in)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def run_epoch(model, loader, pad_id, device, optimizer=None, max_batches=None, grad_clip=1.0):
    """Train or eval one epoch. optimizer=None means eval."""
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, pad_id, device)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


def show_examples(model, tokenizer, device):
    """Greedy-decode one smoke pair both ways."""
    print("\n=== smoke decode ===")
    english, nepali = SMOKE_PAIRS[0]
    for direction in ("en2ne", "ne2en"):
        source, _target = format_pair(english, nepali, direction)
        src_ids = torch.tensor(
            [tokenizer.encode(source, add_special_tokens=False)],
            dtype=torch.long,
            device=device,
        )
        gen = model.generate(src_ids, max_new_tokens=16)[0].tolist()
        print(direction, source, "->", tokenizer.decode(gen, skip_special_tokens=False))


def main():
    """Train on smoke pairs or an OPUS sample."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = load_tokenizer()
    if args.pairs == "opus":
        pairs = load_opus_sample(max_pairs=args.max_pairs)
        print("using opus pairs:", len(pairs))
    else:
        pairs = SMOKE_PAIRS
        print("using smoke pairs:", len(pairs))
    loaders = build_dataloaders(tokenizer, pairs=pairs, batch_size=args.batch_size)
    ids = loaders["ids"]
    print("ids:", ids)

    model = TranslationModel(
        vocab_size=ids["vocab_size"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_enc=args.n_enc,
        n_dec=args.n_dec,
        pad_id=ids["pad_id"],
        bos_id=ids["bos_id"],
        eos_id=ids["eos_id"],
        dropout=args.dropout,
        use_ffn=args.use_ffn,
        cross_attn=args.cross_attn,
    ).to(device)
    print("params:", count_parameters(model), "cross_attn:", args.cross_attn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"needle_{args.cross_attn}_best.pt"
    best = float("inf")

    print("=== train needle ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, loaders["train"], ids["pad_id"], device,
            optimizer=optimizer, max_batches=args.max_batches, grad_clip=args.grad_clip,
        )
        val_loss = run_epoch(
            model, loaders["val"], ids["pad_id"], device,
            optimizer=None, max_batches=args.max_batches,
        )
        print(f"epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss = run_epoch(
        model, loaders["test"], ids["pad_id"], device,
        optimizer=None, max_batches=args.max_batches,
    )
    print("test_loss:", test_loss)
    show_examples(model, tokenizer, device)


if __name__ == "__main__":
    main()