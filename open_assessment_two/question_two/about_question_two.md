# Question 2 — Retrain the UNet Autoencoder on the Celebrity dataset

Open Assessment 2, AI-CR-601. 6 marks.

> Retrain the Unet Autoencoder on the Celebrity dataset.

---

## Understanding

Take the UNet Autoencoder from Question 1 and retrain it on a face dataset (CelebA / Celebrity faces).

- Same architecture idea (encoder–decoder + skip connections).
- Adapt for colour images (3 channels) and higher resolution.
- Primarily the clean autoencoder (reconstruct faces). Denoising can be added with the same flag if useful for the report.

---

## Dataset

**CelebA** (CelebFaces Attributes) via `torchvision.datasets.CelebA`.

- Original size ≈ 178 × 218.
- Resize to **64 × 64** (power of 2, friendly for UNet downsampling). Optional 128 × 128 later if GPU memory allows.
- Transform: `Resize(64) → CenterCrop(64) → ToTensor()` so pixels in [0, 1].
- Official split: train / valid / test. Use train for training, valid for validation.

Note: CelebA download can be large (~1.3 GB). On Colab it is straightforward; locally it may take time.

---

## Architecture adjustments

Reuse the Q1 blocks (`ConvBlock`, `DownBlock`, `UpBlock`, `UNetAE`).

For 64 × 64 with three down stages:
- 64 → 32 → 16 → 8
- Channels still 32 → 64 → 128 (or raise base_ch to 64 if desired)
- Final Sigmoid, 3 output channels

The existing `UpBlock` already handles minor size mismatches via centre-crop, so 64 works cleanly.

---

## Training

- Same loss: MSE
- Adam lr=1e-3 (or 1e-4 if unstable on faces)
- Batch size 64–128 (depends on resolution and GPU)
- Device: `cuda` if available else `cpu`
- Checkpoints: `checkpoints/unet_celeba_ae_best.pt`
- Visual grid: original | reconstructed (and noisy | recon if DAE mode)

---

## Directory structure

```text
open_assessment_two/question_two/
  about_question_two.md
  model.py          # copy/adapt of Q1 UNetAE (or import if path allows)
  main.py           # CelebA loaders + train loop
  code_test.ipynb   # shape checks + sample plots
  checkpoints/
  data/             # CelebA download
```

We keep a local copy of the model so the question is self-contained (same style as OA1).

---

## Workflow

1. Plan here.
2. Implement model + main.
3. Local smoke (adl-cpu):
   ```bash
   python -u main.py --mode ae --epochs 1 --max-batches 10
   ```
4. Full train on Colab T4:
   ```python
   !git clone https://github.com/girisiman/advanced-deep-learning
   %cd advanced-deep-learning/open_assessment_two/question_two
   !python -u main.py --mode ae --epochs 20 --batch-size 64 --img-size 64
   ```

---

## Decisions locked

- Dataset: CelebA, resized to 64×64, 3 channels
- Reuse UNetAE style from Q1
- MSE + Adam
- Primary mode: clean AE (`--mode ae`); DAE optional
- Self-contained files in question_two/
