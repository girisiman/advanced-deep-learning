from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

QUESTION_ONE = Path(__file__).resolve().parent.parent / "question_one"
Q1_DATASET_FILE = QUESTION_ONE / "dataset.py"

spec = importlib.util.spec_from_file_location("q1_dataset", Q1_DATASET_FILE)
q1_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1_dataset)

download_corpus = q1_dataset.download_corpus
tokenize = q1_dataset.tokenize
DATA_FILE = q1_dataset.DATA_FILE

from prompts import (
    SPECIAL_PROMPT_TOKENS,
    format_fill,
    format_next,
    format_translate,
)

CONTEXT_LENGTHS = (4, 8, 12, 16)
MAX_LEN = 48

def load_shakespeare_tokens() -> list[str]:
    """Download Tiny Shakespeare if needed and tokenize it.

    Returns:
        Word tokens from the play.
    """
    download_corpus()
    text = DATA_FILE.read_text(encoding="utf-8")
    return tokenize(text=text)


def make_next_examples(
    tokens: list[str],
    lengths: tuple[int, ...] = CONTEXT_LENGTHS,
) -> list[list[str]]:
    """Windows with a [next] prefix.

    Args:
        tokens: Shakespeare tokens for one split.
        lengths: Window sizes 4-16.

    Returns:
        List of token sequences from format_next.
    """
    examples = []
    n_tokens = len(tokens)
    for context_len in lengths:
        if n_tokens < context_len:
            continue
        for start in range(0, n_tokens - context_len + 1, context_len):
            window = tokens[start : start + context_len]
            examples.append(format_next(tokens=window))
    return examples


def make_fill_examples(
    tokens: list[str],
    lengths: tuple[int, ...] = CONTEXT_LENGTHS,
    seed: int = 42,
) -> list[list[str]]:
    """Windows with one hidden word and a [fill] prefix.

    Args:
        tokens: Shakespeare tokens for one split.
        lengths: Window sizes.
        seed: RNG seed for which word to mask.

    Returns:
        List of token sequences from format_fill.
    """
    rng = random.Random(seed)
    examples = []
    n_tokens = len(tokens)
    for context_len in lengths:
        if n_tokens < context_len:
            continue
        for start in range(0, n_tokens - context_len + 1, context_len):
            window = tokens[start : start + context_len]
            mask_index = rng.randint(1, context_len - 2)
            examples.append(format_fill(tokens=window, mask_index=mask_index))
    return examples

BITEXT_PAIRS = [
    ("hello", "नमस्ते"),
    ("good morning", "शुभ प्रभात"),
    ("good night", "शुभ रात्री"),
    ("thank you", "धन्यवाद"),
    ("yes", "हो"),
    ("no", "होइन"),
    ("how are you", "तिमीलाई कस्तो छ"),
    ("i am fine", "म ठिक छु"),
    ("what is your name", "तिम्रो नाम के हो"),
    ("my name is ram", "मेरो नाम राम हो"),
    ("i live in kathmandu", "म काठमाडौंमा बस्छु"),
    ("please", "कृपया"),
    ("sorry", "माफ गर्नुहोस्"),
    ("see you tomorrow", "भोलि भेटौँला"),
    ("i love nepal", "म नेपाललाई माया गर्छु"),
    ("this is a book", "यो एउटा किताब हो"),
    ("the water is cold", "पानी चिसो छ"),
    ("where is the school", "स्कूल कहाँ छ"),
    ("she is a teacher", "उनी शिक्षक हुन्"),
    ("he is a student", "ऊ विद्यार्थी हो"),
]

def tokenize_sentence(text: str) -> list[str]:
    """Split a short sentence into words. Keeps Nepali tokens.

    Args:
        text: One sentence.

    Returns:
        Token list.
    """
    return text.strip().split()


def make_translate_examples(
    pairs: list[tuple[str, str]] = None,
) -> list[list[str]]:
    """Make [en2ne] and [ne2en] sequences from sentence pairs.

    Args:
        pairs: List of (english, nepali) strings.

    Returns:
        Prompt token lists from format_translate.
    """
    if pairs is None:
        pairs = BITEXT_PAIRS
    examples = []
    for english, nepali in pairs:
        en_tokens = tokenize_sentence(text=english)
        ne_tokens = tokenize_sentence(text=nepali)
        examples.append(
            format_translate(
                source_tokens=en_tokens,
                target_tokens=ne_tokens,
                direction="en2ne",
            )
        )
        examples.append(
            format_translate(
                source_tokens=ne_tokens,
                target_tokens=en_tokens,
                direction="ne2en",
            )
        )
    return examples

def split_list(items: list, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Contiguous 80/10/10 split.

    Args:
        items: Examples or sentence pairs.
        train_ratio: Train fraction.
        val_ratio: Val fraction.

    Returns:
        (train, val, test)
    """
    n_items = len(items)
    n_train = int(n_items * train_ratio)
    n_val = int(n_items * val_ratio)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def cap_list(items: list, max_size: int, seed: int = 42) -> list:
    """Keep at most max_size items so one task cannot drown the mix.

    Args:
        items: Example list.
        max_size: Maximum count.
        seed: RNG seed.

    Returns:
        Capped list.
    """
    if len(items) <= max_size:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, max_size)

class MixVocab:
    """Word ids for the mixed Q3 corpus."""

    def __init__(self, examples: list[list[str]], min_freq: int = 1) -> None:
        """Build vocab from training examples only.

        Args:
            examples: Train token lists (already prefixed).
            min_freq: Rare-word cutoff.
        """
        freq = {}
        for seq in examples:
            for token in seq:
                freq[token] = freq.get(token, 0) + 1

        specials = ["<pad>", "<unk>"] + list(SPECIAL_PROMPT_TOKENS)
        self.itos = list(specials)
        for token, count in sorted(freq.items()):
            if count >= min_freq and token not in specials:
                self.itos.append(token)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def encode(self, tokens: list[str]) -> list[int]:
        unknown = self.unk_id()
        return [self.stoi.get(token, unknown) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]

class PromptDataset(Dataset):
    """One prompted sequence per item, teacher forcing."""

    def __init__(self, examples: list[list[str]], vocab: MixVocab, max_len: int = MAX_LEN) -> None:
        """Encode and store sequences.

        Args:
            examples: Token lists from the format_* functions.
            vocab: MixVocab fitted on train.
            max_len: Truncate longer sequences.
        """
        self.rows = []
        for seq in examples:
            ids = vocab.encode(tokens=seq)[:max_len]
            if len(ids) < 2:
                continue
            self.rows.append((ids[:-1], ids[1:]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        input_ids, labels = self.rows[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "length": len(input_ids),
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    """Right-pad to the longest item in the batch.

    Args:
        batch: Dataset rows.
        pad_id: <pad> id.

    Returns:
        input_ids, labels, lengths tensors.
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for row, item in enumerate(batch):
        k = item["length"]
        input_ids[row, :k] = item["input_ids"]
        labels[row, :k] = item["labels"]
        lengths[row] = k
    return {"input_ids": input_ids, "labels": labels, "lengths": lengths}

def build_dataloaders(
    batch_size: int = 16,
    max_next: int = 400,
    max_fill: int = 400,
    bitext_repeats: int = 8,
    seed: int = 42,
):
    """Build mixed train/val/test loaders.

    Args:
        batch_size: Loader batch size.
        max_next: Cap on [next] train examples.
        max_fill: Cap on [fill] train examples.
        bitext_repeats: Repeat translation pairs so they are not rare.
        seed: Shuffle seed.

    Returns:
        loaders, vocab, stats
    """
    rng = random.Random(seed)
    shakespeare = load_shakespeare_tokens()
    sh_train, sh_val, sh_test = split_list(shakespeare)

    next_train = cap_list(make_next_examples(sh_train), max_next, seed)
    next_val = cap_list(make_next_examples(sh_val), max(max_next // 8, 20), seed)
    next_test = cap_list(make_next_examples(sh_test), max(max_next // 8, 20), seed)

    fill_train = cap_list(make_fill_examples(sh_train, seed=seed), max_fill, seed)
    fill_val = cap_list(make_fill_examples(sh_val, seed=seed + 1), max(max_fill // 8, 20), seed)
    fill_test = cap_list(make_fill_examples(sh_test, seed=seed + 2), max(max_fill // 8, 20), seed)

    pair_train, pair_val, pair_test = split_list(BITEXT_PAIRS)
    tr_train = make_translate_examples(pair_train) * bitext_repeats
    tr_val = make_translate_examples(pair_val)
    tr_test = make_translate_examples(pair_test)

    train_ex = next_train + fill_train + tr_train
    val_ex = next_val + fill_val + tr_val
    test_ex = next_test + fill_test + tr_test
    rng.shuffle(train_ex)

    vocab = MixVocab(examples=train_ex, min_freq=1)
    train_ds = PromptDataset(train_ex, vocab)
    val_ds = PromptDataset(val_ex, vocab)
    test_ds = PromptDataset(test_ex, vocab)

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_batch(batch, vocab.pad_id()),
        )

    loaders = {
        "train": make_loader(train_ds, True),
        "val": make_loader(val_ds, False),
        "test": make_loader(test_ds, False),
    }
    stats = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_next_train": len(next_train),
        "n_fill_train": len(fill_train),
        "n_bitext_train": len(tr_train),
        "vocab_size": len(vocab),
    }
    return loaders, vocab, stats