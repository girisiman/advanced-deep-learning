# Question 4 — Cactus for Text↔Image on Flickr30k

Open Assessment 2, AI-CR-601. 10 marks.

> Use the cactus model (add your own variation for improvement) to do text to image and image to text. For this, use the Flickr 30k dataset.

---

## Goal

Use the Cactus (Needle-style) encoder–decoder from Q3 for two directions on Flickr30k:

- **Image → Text** (captioning)
- **Text → Image** (reconstruction / generation of image patches)

Add a measurable variation for improvement and compare against the pure baseline.

---

## Data

- Source: Hugging Face Flickr30k (`nlphuji/flickr30k` or a parquet mirror such as `AnyModal/flickr30k` / `mosama/flickr_30k`).
- Avoids the Google Drive / gdown problems we hit with CelebA.
- Images resized to **64×64**.
- Text: simple word-level tokenizer built from the training captions (small controlled vocab). Special tokens: `<pad> <unk> <bos> <eos>`.

---

## Model — Approach A (shared Cactus)

One backbone, two modes:

```text
Image → Text:
  image → PatchEmbed → Cactus encoder → memory
  caption tokens → Cactus decoder (cross-attn to image memory) → next-token CE loss

Text → Image:
  caption tokens → embedding → Cactus encoder → memory
  image patches → Cactus decoder (cross-attn to text memory) → MSE on pixels / patches
```

- Reuse pure Cactus blocks from Q3 (`ZCRMSNorm`, RoPE, GQA, gated residual, EncoderBlock, DecoderBlock, NeedleBackbone).
- Image path: `PatchEmbed` + reconstruction head (same as Q3).
- Text path: `nn.Embedding` + linear vocab head.

---

## Variation for improvement (to be proved)

Train and compare two settings:

| Variant | Description |
|---------|-------------|
| **Baseline** | Pure Cactus (no FFN, cross-attention on **all** decoder layers) |
| **Improved** | One of: (a) optional FFN turned on, or (b) cross-attention only on the **last** decoder layer |

Metrics (same for both):
- Image→Text: cross-entropy / simple token accuracy or BLEU-like score
- Text→Image: MSE

Report the numbers so the improvement is visible.

---

## Directory

```text
open_assessment_two/question_four/
  about_question_four.md
  data.py            # HF Flickr30k loader + word tokenizer
  needle.py          # Cactus blocks (from Q3 or imported)
  model.py           # multimodal Cactus + variation flag
  main.py            # train baseline / improved
  code_test.ipynb
```

---

## Workflow

1. Plan (this file).
2. `data.py` — lock the Hugging Face loader and tokenizer.
3. `model.py` — shared Cactus with `mode="i2t"|"t2i"` and a variation flag.
4. `main.py` — training loop that can run either variant.
5. Local smoke (few batches) → full train on Colab.
6. Compare baseline vs improved and write the numbers.

---

## Decisions locked

- Dataset: Hugging Face Flickr30k
- Image size: 64×64
- Approach: shared Cactus (Approach A)
- Improvement: measurable (FFN or 1-vs-all cross-attn)
- Pure Cactus blocks reused from Q3
