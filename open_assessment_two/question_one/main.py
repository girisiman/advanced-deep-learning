"""
Train a UNet Autoencoder (and Denoising Autoencoder) on MNIST.

Usage (local smoke test, adl-cpu):
  python -u main.py --mode ae  --epochs 1 --max-batches 20
  python -u main.py --mode dae --epochs 1 --max-batches 20

Full train (Colab):
  !python main.py --mode ae  --epochs 15
  !python main.py --mode dae --epochs 15
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from model import UNetAE, add_gaussian_noise


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(batch_size: int, val_frac: float = 0.1, data_root: str = "./data"):
    tfm = transforms.ToTensor()  # [0, 1]
    full_train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=tfm)

    n_val = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optim, device, mode: str, sigma: float, max_batches: int = 0):
    model.train()
    criterion = nn.MSELoss()
    total_loss, n = 0.0, 0
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        target = x
        inp = add_gaussian_noise(x, sigma) if mode == "dae" else x

        optim.zero_grad()
        recon = model(inp)
        loss = criterion(recon, target)
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, mode: str, sigma: float, max_batches: int = 0):
    model.eval()
    criterion = nn.MSELoss()
    total_loss, n = 0.0, 0
    for i, (x, _) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device)
        target = x
        inp = add_gaussian_noise(x, sigma) if mode == "dae" else x
        recon = model(inp)
        loss = criterion(recon, target)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def save_reconstructions(model, loader, device, mode: str, sigma: float, out_path: Path, n_show: int = 8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n_show].to(device)
    inp = add_gaussian_noise(x, sigma) if mode == "dae" else x
    recon = model(inp)
    # original | input (noisy if dae) | recon
    grid = make_grid(torch.cat([x, inp, recon], dim=0), nrow=n_show)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"saved reconstructions to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ae", "dae"], default="ae")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=0.2, help="Gaussian noise std for DAE")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=0, help="0 = full epoch (smoke-test helper)")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"device={device} mode={args.mode} sigma={args.sigma}")

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, data_root=args.data_root)
    model = UNetAE(in_channels=1, base_ch=args.base_ch).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = ckpt_dir / f"unet_{args.mode}_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device, args.mode, args.sigma, args.max_batches)
        va_loss = evaluate(model, val_loader, device, args.mode, args.sigma, args.max_batches)
        print(f"epoch {epoch:02d}  train_mse={tr_loss:.5f}  val_mse={va_loss:.5f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mse": va_loss, "args": vars(args)}, best_path)
            print(f"  saved best → {best_path}")

    # final test + visual
    test_mse = evaluate(model, test_loader, device, args.mode, args.sigma)
    print(f"test_mse={test_mse:.5f}")
    save_reconstructions(
        model, test_loader, device, args.mode, args.sigma,
        Path(f"recon_{args.mode}.png"),
    )


if __name__ == "__main__":
    main()
