# Question 1 — UNet Autoencoder + Denoising Autoencoder on MNIST

Open Assessment 2, AI-CR-601. 6 marks.

> Implement an Unet Autoencoder on the MNIST. Also train a Denoising Autoencoder on the same dataset.

---

## Understanding

Build two related models on MNIST (1×28×28 grayscale digits):

1. **UNet Autoencoder** — reconstruct clean images from clean inputs. Encoder–decoder with skip connections.
2. **Denoising Autoencoder** — same architecture, trained to map noisy images → clean images.

Same model class. Only the training input changes (clean vs noisy). Loss is pixel-wise MSE in both cases.

Skip connections (concatenation) make it a UNet rather than a plain bottleneck autoencoder.

---

## Architecture

Lightweight convolutional UNet:

- Input: 1 × 28 × 28
- Encoder: three DownBlocks (ConvBlock + MaxPool)
  - Channels: 32 → 64 → 128
  - Spatial: 28 → 14 → 7
- Bottleneck: ConvBlock at 7×7 (128 channels)
- Decoder: two UpBlocks (ConvTranspose + concat skip + ConvBlock) + final 1×1 conv
  - Channels back to 32 then 1
  - Spatial back to 28×28
- Output: Sigmoid (pixels in [0, 1])

28 is not a power of 2, so UpBlock centres-crops the skip if sizes differ.

---

## Denoising

- Training only: `noisy = clamp(clean + σ * randn_like(clean), 0, 1)`
- Default σ = 0.2
- Target is always the clean image

---

## Dataset & training

- `torchvision.datasets.MNIST`, `ToTensor()` → [0, 1]
- Official train set split 90/10 train/val; official test held out
- Batch size 128, Adam lr=1e-3
- Device: `cuda` if available else `cpu`
- Checkpoints: `checkpoints/unet_ae_best.pt`, `checkpoints/unet_dae_best.pt`
- Visual grid: original | input | reconstructed

---

## Directory structure

```text
open_assessment_two/question_one/
  about_question_one.md
  model.py          # UNetAE, ConvBlock, DownBlock, UpBlock, add_gaussian_noise
  main.py           # data loaders, train, eval, both modes
  code_test.ipynb   # stepwise shape and smoke tests
  checkpoints/      # gitignored
  data/             # MNIST download, gitignored
```

---

## Workflow (same as OA1)

1. Plan here.
2. Write one function / class at a time with clear signatures and comments.
3. Test in `code_test.ipynb`.
4. Local smoke (adl-cpu only, prompt must be only `(adl-cpu)`):

   ```bash
   python -u main.py --mode ae  --epochs 1 --max-batches 20
   python -u main.py --mode dae --epochs 1 --max-batches 20
   ```

5. Full train on Colab T4 (clone repo, no pip install of requirements.txt):

   ```python
   !git clone https://github.com/girisiman/advanced-deep-learning
   %cd advanced-deep-learning/open_assessment_two/question_one
   !python -u main.py --mode ae  --epochs 15
   !python -u main.py --mode dae --epochs 15
   ```

---

## Decisions locked

- One model class `UNetAE`, controlled by `--mode ae|dae`
- Channel progression 32-64-128
- MSE + Adam 1e-3
- Fixed Gaussian noise σ=0.2 for DAE
- Sigmoid output, pixels [0, 1]
- No custom dataset.py (MNIST is standard torchvision)
