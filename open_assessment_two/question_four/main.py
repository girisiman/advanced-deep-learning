"""
Train Cactus multimodal model on Flickr30k.

Variants:
  python main.py --variant baseline
  python main.py --variant ffn
  python main.py --variant cross_last
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_train_captions, build_vocab, Flickr30kDataset
from model import CactusMultiModal


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "ffn", "cross_last"], default="baseline")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None, help="limit dataset size for smoke tests")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--dataset", default="AnyModal/flickr30k")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    return p.parse_args()


def variant_to_flags(name):
    if name == "baseline":
        return False, "all"
    if name == "ffn":
        return True, "all"
    if name == "cross_last":
        return False, "last"
    raise ValueError(name)


def main():
    args = parse_args()
    device = get_device()
    use_ffn, cross_attn = variant_to_flags(args.variant)
    print(f"device={device}  variant={args.variant}  use_ffn={use_ffn}  cross_attn={cross_attn}")

    # vocab
    captions = get_train_captions(dataset_name=args.dataset, max_samples=args.max_samples)
    stoi, itos = build_vocab(captions, min_freq=2)
    vocab_size = len(itos)
    print(f"vocab size: {vocab_size}")

    # data
    train_ds = Flickr30kDataset(
        split="train", img_size=args.img_size, max_len=args.max_len,
        stoi=stoi, dataset_name=args.dataset, max_samples=args.max_samples,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # model
    model = CactusMultiModal(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        img_size=args.img_size,
        use_ffn=use_ffn,
        cross_attn=cross_attn,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ce = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])
    mse = nn.MSELoss()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"cactus_{args.variant}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_i2t, total_t2i, n = 0.0, 0.0, 0
        for i, (img, ids) in enumerate(train_loader):
            if args.max_batches and i >= args.max_batches:
                break
            img, ids = img.to(device), ids.to(device)

            # image → text
            logits = model(img, ids, mode="i2t")
            loss_i2t = ce(logits.view(-1, vocab_size), ids.view(-1))

            # text → image
            recon = model(img, ids, mode="t2i")
            loss_t2i = mse(recon, img)

            loss = loss_i2t + loss_t2i
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_i2t += loss_i2t.item() * img.size(0)
            total_t2i += loss_t2i.item() * img.size(0)
            n += img.size(0)

        print(f"epoch {epoch:02d}  i2t_ce={total_i2t/n:.4f}  t2i_mse={total_t2i/n:.4f}")
        torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "args": vars(args)}, best_path)

    print(f"saved → {best_path}")


if __name__ == "__main__":
    main()