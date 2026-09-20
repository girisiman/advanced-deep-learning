# Question 1 — Understanding, Plan, Dataset

Open Assessment 1, AI-CR-601. 8 marks.

> Implement RNN and LSTM from scratch in PyTorch. Use the LSTM for next-word prediction (sequence length 4–16). Evaluate on unseen test data.

---

## Understanding

Three things to build:

1. Vanilla RNN from scratch (no `nn.RNN` / `nn.LSTM`)
2. LSTM from scratch (same interface)
3. Next-word model that uses **my** LSTM, context 4–16 words, scores on held-out text

From scratch = write the time loop and gates with `nn.Linear`, `sigmoid`, `tanh`. Autograd is fine.

**Next word:** given k words, predict word k+1. Example: `the cat sat on` → `the`.

**RNN:** h_t = tanh(W_x x_t + W_h h_{t-1} + b). Hidden state is overwritten every step. Long windows (k=16) forget — that is the point of the lecture.

**LSTM:** extra cell c_t plus three gates.

- forget f_t: keep or drop old cell
- input i_t: write new candidate
- output o_t: how much of the cell becomes h_t

c_t = f_t * c_{t-1} + i_t * candidate

h_t = o_t * tanh(c_t)

Same data, RNN vs LSTM. LSTM should be better at k=16.

Q2 will swap this backbone for a Transformer, so keep:

`backbone(x) -> outputs`

---

## Dataset

**Tiny Shakespeare**, word-level. Small, public domain, CPU-friendly. Q4 datasets (OPUS, FLORES, Gemma/Qwen) stay out of Q1.

- lowercase, split into words
- vocab from train only: `<pad> <unk> <bos> <eos>`
- split the file 80 / 10 / 10 (train / val / test). Do not shuffle windows or test leaks
- windows of length 4, 8, 12, 16
- train with teacher forcing; report next-word accuracy per length

---

## Plan

1. `dataset.py` — download, tokenize, windows
2. `model.py` — RNNCell, LSTMCell, NextWordModel
3. Train LSTM, then RNN on the same setup
4. Test: perplexity, top-1 / top-5, acc at k=4 vs k=16, a few generated sentences
5. Write the numbers and whether LSTM won at length 16

```text
question_one/
  about_question_one.md
  main.py
  model.py
  dataset.py
  utils.py
  data/          # gitignored