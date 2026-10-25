# Question 2 — Understanding, Plan, Dataset

Open Assessment 1, AI-CR-601. 8 marks.

> Replace the default Recurrent block with your own vanilla Transformer implementation.

---

## Understanding

Q1 already has next-word prediction (context 4–16) with my RNN and LSTM. Q2 does not change the task. It replaces the recurrent backbone with a Transformer I write myself.

No `nn.Transformer` / `nn.MultiheadAttention`. Allowed: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`.

For next-word prediction the block must be **causal**. Position t may attend to tokens 1…t only. If it can see the future, the test numbers are invalid.

Same interface as Q1 so train/eval stay unchanged:

```text
output, extra = backbone(x)    # x: (batch, time, embed)