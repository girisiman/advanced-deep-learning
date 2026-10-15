# Question 1 — UNet Autoencoder + Denoising Autoencoder on MNIST

Open Assessment 2, AI-CR-601. 6 marks.

> Implement an Unet Autoencoder on the MNIST. Also train a Denoising Autoencoder on the same dataset.

---

## Understanding

Two related models on MNIST (28×28 grayscale digits):

1. **UNet Autoencoder**: reconstruct clean images from clean inputs. Encoder–decoder with skip connections (the UNet part).
2. **Denoising Autoencoder**: same architecture, but trained to map noisy images → clean images.

Same model class; only the training input changes. Loss is pixel-wise MSE in both cases.

Skip connections (concatenation) are required so it is recognisably a UNet, not a plain bottleneck autoencoder.

---

## Architecture (lightweight convolutional UNet)

- Input: 1 × 28 × 28
- Encoder: three down blocks (Conv + ReLU + optional BatchNorm, then MaxPool)
  - Channels: 32 → 64 → 128
  - Spatial: 28 → 14 → 7
- Bottleneck: 128-channel feature map at 7×7
- Decoder: three up blocks (upsample/ConvTranspose + Conv + ReLU) with skip concatenations
  - Channels: 128 → 64 → 32 → 1
  - Spatial back to 28×28
- Final activation: Sigmoid (pixels kept in [0, 1])

Padding/cropping is handled so the final spatial size matches exactly 28×28 (28 is not a power of 2).

---

## Denoising

- Training only: `noisy = clamp(clean + σ * randn_like(clean), 0, 1)`
- Default σ = 0.2 (fixed; can be made random later)
- Target is always the clean image
- Inference: can take either clean or noisy input

---

## Dataset & training

- torchvision MNIST, download on first run
- Pixels scaled to [0, 1]
- Train/val split of the official training set (e.g. 90/10); official test set held out for final evaluation
- Batch size 64–128, Adam lr=1e-3
- Device: `cuda` if available else `cpu`
- Checkpoints: `checkpoints/unet_ae_best.pt`, `checkpoints/unet_dae_best.pt`
- Visualisation: grid of original / (noisy) / reconstructed

---

## Files

```text
open_assessment_two/question_one/
  about_question_one.md
  model.py          # UNetAE + building blocks
  main.py           # data, train, eval, both modes
  code_test.ipynb   # stepwise tests
  checkpoints/      # gitignored
```

Smoke test (local, adl-cpu):

```bash
python -u main.py --mode ae --epochs 1 --max-batches 20
python -u main.py --mode dae --epochs 1 --max-batches 20
```

Full train on Colab T4: clone repo, `cd open_assessment_two/question_one`, `!python main.py --mode ae --epochs 15` (and same for dae).

---

## Decisions locked for now

- One model class `UNetAE`, flag `--mode ae|dae`
- Channel progression 32-64-128
- MSE loss, Adam 1e-3
- Fixed Gaussian noise σ=0.2 for DAE
- Sigmoid output, pixels [0, 1]
