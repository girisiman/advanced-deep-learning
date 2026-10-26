"""
Train the Cactus image autoencoder on CelebA.

Local smoke:
  python -u main.py --epochs 1 --max-batches 5

Colab:
  !python -u main.py --epochs 15 --batch-size 32
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from model import CactusImageAE


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(batch_size: int, img_size: int = 64, data_root: str = "./data"):
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    train_set = datasets.CelebA(data_root, split="train", download=True, transform=tfm)
    val_set = datasets.CelebA(data_root, split="valid", download=True, transform=tfm)
    test_set = datasets.CelebA(data_root, split="test", download=True, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optim, device, max_batches: int = 0):
    model.train()
    criterion = nn.MSELoss()
    total, n = 0.0, 0
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        optim.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 0):
    model.eval()
    criterion = nn.MSELoss()
    total, n = 0.0, 0
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        recon = model(x)
        loss = criterion(recon, x)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def save_reconstructions(model, loader, device, out_path: Path, n_show: int = 8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n_show].to(device)
    recon = model(x)
    grid = make_grid(torch.cat([x, recon], dim=0), nrow=n_show)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-encoder-layers", type=int, default=3)
    p.add_argument("--n-decoder-layers", type=int, default=3)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"device={device}")

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, args.img_size, args.data_root)
    model = CactusImageAE(
        img_size=args.img_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = ckpt_dir / "cactus_ae_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device, args.max_batches)
        va = evaluate(model, val_loader, device, args.max_batches)
        print(f"epoch {epoch:02d}  train_mse={tr:.5f}  val_mse={va:.5f}")
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mse": va, "args": vars(args)}, best_path)
            print(f"  saved → {best_path}")

    test_mse = evaluate(model, test_loader, device)
    print(f"test_mse={test_mse:.5f}")
    save_reconstructions(model, test_loader, device, Path("recon_cactus.png"))


if __name__ == "__main__":
    main()