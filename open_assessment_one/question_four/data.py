"""EN↔NE bitext for Question 4."""

from __future__ import annotations
from datasets import load_dataset

PREFIX_EN2NE = "[en2ne]"
PREFIX_NE2EN = "[ne2en]"

SMOKE_PAIRS: list[tuple[str, str]] = [
    ("good morning", "शुभ प्रभात"),
    ("thank you", "धन्यवाद"),
    ("how are you", "तपाईंलाई कस्तो छ"),
    ("I am fine", "म ठिक छु"),
    ("see you tomorrow", "भोलि भेटौँला"),
]


def format_pair(english: str, nepali: str, direction: str) -> tuple[str, str]:
    """Turn one bitext row into source and target strings.

    Args:
        english: English sentence.
        nepali: Nepali sentence.
        direction: 'en2ne' or 'ne2en'.

    Returns:
        (source, target) strings.
    """
    english = english.strip()
    nepali = nepali.strip()
    if not english or not nepali:
        raise ValueError("both sides of a pair must be non-empty")
    if direction == "en2ne":
        return f"{PREFIX_EN2NE} {english}", nepali
    if direction == "ne2en":
        return f"{PREFIX_NE2EN} {nepali}", english
    raise ValueError("direction must be 'en2ne' or 'ne2en'")


def expand_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Duplicate each pair into both directions.

    Args:
        pairs: List of (english, nepali).

    Returns:
        List of (source, target, direction).
    """
    rows: list[tuple[str, str, str]] = []
    for english, nepali in pairs:
        src, tgt = format_pair(english, nepali, "en2ne")
        rows.append((src, tgt, "en2ne"))
        src, tgt = format_pair(english, nepali, "ne2en")
        rows.append((src, tgt, "ne2en"))
    return rows

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"
BOS_TOKEN = "[bos]"
DIRECTION_TOKENS = (PREFIX_EN2NE, PREFIX_NE2EN)
EXTRA_TOKENS = (PREFIX_EN2NE, PREFIX_NE2EN, BOS_TOKEN)


def load_tokenizer(name: str = DEFAULT_TOKENIZER):
    """Load one tokenizer for English and Nepali.

    Only tokenizer files are pulled, not Qwen weights.

    Args:
        name: Hugging Face tokenizer id.

    Returns:
        A transformers tokenizer.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Need transformers. On adl-cpu: pip install transformers. "
            "On Colab do not pip install requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    extra = [tok for tok in EXTRA_TOKENS if tok not in tokenizer.get_vocab()]
    if extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
    return tokenizer


def special_ids(tokenizer) -> dict[str, int]:
    """Pad / BOS / EOS ids the model needs.

    Args:
        tokenizer: Shared tokenizer.

    Returns:
        Dict with pad_id, bos_id, eos_id, vocab_size.
    """
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id)
    bos_id = int(tokenizer.convert_tokens_to_ids(BOS_TOKEN))
    return {
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "vocab_size": len(tokenizer),
    }


def encode_text(tokenizer, text: str) -> list[int]:
    """Encode one string, no extra BOS/EOS.

    Args:
        tokenizer: Shared tokenizer.
        text: Source or target string.

    Returns:
        Token ids.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def encode_pair(tokenizer, source: str, target: str) -> tuple[list[int], list[int]]:
    """Encode one formatted pair.

    Args:
        tokenizer: Shared tokenizer.
        source: Prefixed source string.
        target: Target string.

    Returns:
        (src_ids, tgt_ids) with BOS/EOS on the target.
    """
    ids = special_ids(tokenizer)
    src_ids = encode_text(tokenizer, source)
    tgt_ids = [ids["bos_id"]] + encode_text(tokenizer, target) + [ids["eos_id"]]
    return src_ids, tgt_ids

def _pad_2d(rows: list[list[int]], pad_id: int):
    """Pad variable-length id lists into a rectangle.

    Args:
        rows: List of token-id lists.
        pad_id: Pad id.

    Returns:
        Long tensor (batch, max_len).
    """
    import torch

    if not rows:
        raise ValueError("cannot pad an empty batch")
    max_len = max(len(r) for r in rows)
    out = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate_batch(
    batch: list[tuple[list[int], list[int]]],
    pad_id: int,
) -> dict:
    """Pad source and target ids.

    tgt_in is teacher-force input (drop last).
    labels is the next-token target (drop first).

    Args:
        batch: List of (src_ids, tgt_ids).
        pad_id: Tokenizer pad id.

    Returns:
        Dict with src_ids, tgt_in, labels.
    """
    src_ids = _pad_2d([src for src, _ in batch], pad_id)
    tgt_ids = _pad_2d([tgt for _, tgt in batch], pad_id)
    return {
        "src_ids": src_ids,
        "tgt_in": tgt_ids[:, :-1],
        "labels": tgt_ids[:, 1:],
    }

def encode_rows(tokenizer, pairs, max_src=96, max_tgt=96):
    """Expand pairs both ways and encode them.

    Args:
        tokenizer: Shared tokenizer.
        pairs: List of (english, nepali).
        max_src: Truncate source to this length.
        max_tgt: Truncate target to this length.

    Returns:
        List of (src_ids, tgt_ids).
    """
    rows = []
    for source, target, _direction in expand_pairs(pairs):
        src_ids, tgt_ids = encode_pair(tokenizer, source, target)
        src_ids = src_ids[:max_src]
        tgt_ids = tgt_ids[:max_tgt]
        if len(src_ids) < 1 or len(tgt_ids) < 2:
            continue
        rows.append((src_ids, tgt_ids))
    return rows


def split_rows(rows, seed=0):
    """Contiguous 80/10/10 split after a seeded shuffle.

    Args:
        rows: Encoded pairs.
        seed: Shuffle seed.

    Returns:
        Dict with train, val, test lists.
    """
    import random

    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :] or val[-1:]
    return {"train": train, "val": val, "test": test}


class BitextDataset:
    """List of (src_ids, tgt_ids).

    Args:
        rows: Encoded pairs.
    """

    def __init__(self, rows):
        if not rows:
            raise ValueError("dataset is empty")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def build_dataloaders(tokenizer, pairs=None, batch_size=4, max_src=96, max_tgt=96, seed=0):
    """Encode pairs and wrap them in DataLoaders.

    Args:
        tokenizer: Shared tokenizer.
        pairs: (english, nepali) rows. Defaults to SMOKE_PAIRS.
        batch_size: Batch size.
        max_src: Source truncate length.
        max_tgt: Target truncate length.
        seed: Split seed.

    Returns:
        Dict of train/val/test loaders plus special ids.
    """
    from torch.utils.data import DataLoader

    if pairs is None:
        pairs = SMOKE_PAIRS
    ids = special_ids(tokenizer)
    split = split_rows(encode_rows(tokenizer, pairs, max_src, max_tgt), seed=seed)

    def _collate(batch):
        return collate_batch(batch, pad_id=ids["pad_id"])

    loaders = {
        name: DataLoader(
            BitextDataset(rows),
            batch_size=batch_size,
            shuffle=(name == "train"),
            collate_fn=_collate,
        )
        for name, rows in split.items()
    }
    loaders["ids"] = ids
    return loaders


from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OPUS_CACHE = DATA_DIR / "opus_en_ne_sample.jsonl"
OPUS_DATASET = "Helsinki-NLP/opus-100"
OPUS_CONFIG = "en-ne"
DEFAULT_SAMPLE_SIZE = 8000


import re

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def _pair_ok(english: str, nepali: str) -> bool:
    """Drop empty, identical, tiny, or script-mismatched pairs."""
    english = english.strip()
    nepali = nepali.strip()
    if len(english) < 8 or len(nepali) < 8:
        return False
    if english == nepali:
        return False
    if not LATIN.search(english):
        return False
    if not DEVANAGARI.search(nepali):
        return False
    return True


def _write_jsonl(path, pairs):
    """Save (english, nepali) rows as JSONL."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for english, nepali in pairs:
            handle.write(json.dumps({"en": english, "ne": nepali}, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    """Load cached (english, nepali) rows."""
    import json

    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pairs.append((row["en"], row["ne"]))
    return pairs

def load_opus_sample(max_pairs=DEFAULT_SAMPLE_SIZE, cache_path=OPUS_CACHE, force=False):
    """Download a small OPUS-100 EN-NE sample and cache it.

    Full train split is ~406k pairs. We keep max_pairs only.

    Args:
        max_pairs: How many pairs to keep.
        cache_path: Local JSONL cache.
        force: Ignore cache and download again.

    Returns:
        List of (english, nepali).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        pairs = _read_jsonl(cache_path)
        print("opus cache", cache_path, "n=", len(pairs))
        return pairs[:max_pairs]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Need datasets. On adl-cpu: pip install datasets. "
            "On Colab do not pip install requirements.txt."
        ) from exc

    print("downloading", OPUS_DATASET, OPUS_CONFIG, "max", max_pairs)
    stream = load_dataset(OPUS_DATASET, OPUS_CONFIG, split="train", streaming=True)
    pairs = []
    for row in stream:
        trans = row["translation"]
        english = str(trans.get("en", "")).strip()
        nepali = str(trans.get("ne", "")).strip()
        if _pair_ok(english, nepali):
            pairs.append((english, nepali))
        if len(pairs) >= max_pairs:
            break
    if len(pairs) < 10:
        raise RuntimeError(f"OPUS sample too small: {len(pairs)}")
    _write_jsonl(cache_path, pairs)
    print("opus saved", cache_path, "n=", len(pairs))
    return pairs

FLORES_CACHE = DATA_DIR / "flores_en_ne_devtest.jsonl"
FLORES_DATASET = "facebook-llama/flores"
FLORES_CONFIG = "neen"
FLORES_SPLIT = "test"


from datasets import load_dataset

FLORES_CACHE = DATA_DIR / "flores_en_ne_devtest.jsonl"
FLORES_TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def load_flores(split="devtest", max_pairs=None, cache_path=None, force=False):
    """Load FLORES-200 English–Nepali pairs without a gated Hub dataset.

    Downloads the official tarball once, then reads
    eng_Latn and npi_Deva files. Same sentence index = same sentence.

    Args:
        split: 'dev' or 'devtest'.
        max_pairs: Optional cap.
        cache_path: Local JSONL cache.
        force: Ignore cache.

    Returns:
        List of (english, nepali).
    """
    import tarfile
    import tempfile
    import urllib.request

    if cache_path is None:
        cache_path = FLORES_CACHE
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        pairs = _read_jsonl(cache_path)
        print("flores cache", cache_path, "n=", len(pairs))
        return pairs[:max_pairs] if max_pairs else pairs

    print("downloading FLORES-200 tarball")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "flores200_dataset.tar.gz"
    if not tar_path.exists() or force:
        urllib.request.urlretrieve(FLORES_TAR_URL, tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        en_name = f"flores200_dataset/{split}/eng_Latn.{split}"
        ne_name = f"flores200_dataset/{split}/npi_Deva.{split}"
        en_lines = tar.extractfile(en_name).read().decode("utf-8").splitlines()
        ne_lines = tar.extractfile(ne_name).read().decode("utf-8").splitlines()

    pairs = []
    for english, nepali in zip(en_lines, ne_lines):
        english, nepali = english.strip(), nepali.strip()
        if english and nepali:
            pairs.append((english, nepali))
    _write_jsonl(cache_path, pairs)
    print("flores saved", cache_path, "n=", len(pairs))
    return pairs[:max_pairs] if max_pairs else pairs
