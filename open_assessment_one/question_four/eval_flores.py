"""Score a Q4 checkpoint on FLORES-200 EN↔NE.

Run from question_four after a Colab train:

    python -u eval_flores.py --ckpt checkpoints/needle_all_best.pt --max-pairs 50
    python -u eval_flores.py --ckpt checkpoints/needle_one_best.pt --max-pairs 50
    python -u eval_flores.py --ckpt checkpoints/needle_all_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import format_pair, load_flores, load_tokenizer, special_ids
from model import TranslationModel


def parse_args() -> argparse.Namespace:
    """Command-line settings."""
    parser = argparse.ArgumentParser(description="Q4 FLORES eval")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-enc", type=int, default=2)
    parser.add_argument("--n-dec", type=int, default=2)
    parser.add_argument("--cross-attn", choices=("all", "one"), default=None)
    parser.add_argument("--use-ffn", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def infer_cross_attn(ckpt: Path, flag: str | None) -> str:
    """Read all/one from the filename if the user omitted --cross-attn."""
    if flag is not None:
        return flag
    name = ckpt.name.lower()
    if "one" in name:
        return "one"
    return "all"


def strip_special(text: str) -> str:
    """Drop BOS / direction leftovers from a decode."""
    for tok in ("[bos]", "[en2ne]", "[ne2en]", "<|endoftext|>"):
        text = text.replace(tok, " ")
    return " ".join(text.split())


def translate_one(model, tokenizer, source: str, device, max_new_tokens: int) -> str:
    """Greedy-decode one source string."""
    src_ids = torch.tensor(
        [tokenizer.encode(source, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    gen = model.generate(src_ids, max_new_tokens=max_new_tokens)[0].tolist()
    return strip_special(tokenizer.decode(gen, skip_special_tokens=False))


def score(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Corpus BLEU and chrF."""
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError("Need sacrebleu. On Colab: pip install sacrebleu") from exc
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


def main() -> None:
    """Load FLORES, translate both ways, print BLEU/chrF."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    cross_attn = infer_cross_attn(ckpt, args.cross_attn)
    print("device:", device, "ckpt:", ckpt, "cross_attn:", cross_attn)

    tokenizer = load_tokenizer()
    ids = special_ids(tokenizer)
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
        use_ffn=args.use_ffn,
        cross_attn=cross_attn,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pairs = load_flores(max_pairs=args.max_pairs)
    print("flores pairs:", len(pairs))

    for direction in ("en2ne", "ne2en"):
        hyps: list[str] = []
        refs: list[str] = []
        for i, (english, nepali) in enumerate(pairs):
            source, target = format_pair(english, nepali, direction)
            hyp = translate_one(model, tokenizer, source, device, args.max_new_tokens)
            hyps.append(hyp)
            refs.append(target)
            if i < 2:
                print(direction, "src:", source)
                print(direction, "ref:", target)
                print(direction, "hyp:", hyp)
        metrics = score(hyps, refs)
        print(f"{direction}  bleu={metrics['bleu']:.2f}  chrf={metrics['chrf']:.2f}")


if __name__ == "__main__":
    main()
