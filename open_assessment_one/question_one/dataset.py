"""Tiny Shakespeare next-word dataset for Question 1.

Word-level tokens, contiguous 80/10/10 split, context lengths 4, 8, 12 and 16.

Run from this folder:
    python dataset.py
"""
from __future__ import annotations

import random
import re
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"

# Exam asks for sequence length 4-16. These four sizes cover that range.
CONTEXT_LENGTHS = (4, 8, 12, 16)

# Special tokens. <pad> must stay at index 0 for easy masking later.
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

# Words like "don't" stay one token. Punctuation is its own token.
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?|[.,!?;:]+"
TOKEN_RE = re.compile(TOKEN_PATTERN)
# ---------------------------------------------------------------------------
# Step 1: download and tokenize
# ---------------------------------------------------------------------------

def download_corpus(url: str = DATA_URL, path: Path = DATA_FILE) -> Path:
    """Download Tiny Shakespeare once and save it under data/.

    Args:
        url: Remote text file.
        path: Local destination.

    Returns:
        Path to the saved text file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    print("Downloading Tiny Shakespeare ->", path)
    urllib.request.urlretrieve(url, path)
    return path


def read_text(path: Path = DATA_FILE) -> str:
    """Read the corpus as a single string.

    Args:
        path: Local text file.

    Returns:
        File contents.
    """
    return path.read_text(encoding="utf-8")


def tokenize(text: str, pattern: re.Pattern = TOKEN_RE) -> list[str]:
    """Lowercase the text and split it into word tokens.

    Args:
        text: Raw corpus string.
        pattern: Regex used to find tokens.

    Returns:
        List of tokens, for example ["first", "citizen", ":"].
    """
    # Lowercase first so "The" and "the" share one vocab entry.
    return pattern.findall(text.lower())


# ---------------------------------------------------------------------------
# Step 2: vocabulary (built from the training split only)
# ---------------------------------------------------------------------------

def count_tokens(tokens: list[str]) -> dict[str, int]:
    """Count how often each token appears.

    Args:
        tokens: Token list (training split).

    Returns:
        Mapping token -> frequency.
    """
    freq = {}
    for token in tokens:
        freq[token] = freq.get(token, 0) + 1
    return freq


def build_vocab_list(
    tokens: list[str],
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
    min_freq: int = 2,
) -> list[str]:
    """Build the ordered vocab list: special tokens first, then words.

    Args:
        tokens: Training tokens only. Val/test words must not be added here.
        special_tokens: Always kept, even if they never appear in text.
        min_freq: Drop rare words so they become <unk>.

    Returns:
        List of strings. Index in this list is the token id.
    """
    freq = count_tokens(tokens)
    vocab_list = list(special_tokens)
    for token, count in sorted(freq.items()):
        if count >= min_freq and token not in special_tokens:
            vocab_list.append(token)
    return vocab_list


class Vocab:
    """String <-> integer map used by the model."""

    def __init__(
        self,
        tokens: list[str],
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        min_freq: int = 2,
    ) -> None:
        """Create vocab from training tokens.

        Args:
            tokens: Training tokens.
            special_tokens: <pad>, <unk>, <bos>, <eos>.
            min_freq: Minimum count to keep a word.
        """
        self.itos = build_vocab_list(
            tokens=tokens,
            special_tokens=special_tokens,
            min_freq=min_freq,
        )
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def pad_id(self) -> int:
        """Id used to fill shorter sequences in a batch."""
        return self.stoi["<pad>"]

    def unk_id(self) -> int:
        """Id used for words that are missing from the training vocab."""
        return self.stoi["<unk>"]

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens to ids. Unknown words become <unk>.

        Args:
            tokens: List of word strings.

        Returns:
            List of integer ids.
        """
        unknown = self.unk_id()
        return [self.stoi.get(token, unknown) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert ids back to tokens.

        Args:
            ids: List of integer ids.

        Returns:
            List of word strings.
        """
        words = []
        for token_id in ids:
            if 0 <= token_id < len(self.itos):
                words.append(self.itos[token_id])
            else:
                words.append("<unk>")
        return words


# ---------------------------------------------------------------------------
# Step 3: split and cut windows of length 4-16
# ---------------------------------------------------------------------------

def split_tokens(
    tokens: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """Contiguous split so neighbouring windows cannot leak into test.

    Args:
        tokens: Full tokenized corpus.
        train_ratio: Fraction for training (default 80%).
        val_ratio: Fraction for validation (default 10%).
            The remainder is the test set.

    Returns:
        (train_tokens, val_tokens, test_tokens)
    """
    n_tokens = len(tokens)
    n_train = int(n_tokens * train_ratio)
    n_val = int(n_tokens * val_ratio)
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train : n_train + n_val]
    test_tokens = tokens[n_train + n_val :]
    return train_tokens, val_tokens, test_tokens


def make_windows(
    ids: list[int],
    lengths: tuple[int, ...] = CONTEXT_LENGTHS,
) -> list[tuple[list[int], list[int], int]]:
    """Cut teacher-forcing windows from a token-id sequence.

    For context length k:
        input  x = ids[i : i+k]
        target y = ids[i+1 : i+k+1]
    so each position predicts the next word.

    Args:
        ids: Token ids for one split (train or val or test).
        lengths: Context sizes to build. Exam range is 4-16.

    Returns:
        List of (input_ids, label_ids, k).
    """
    windows = []
    n_ids = len(ids)
    for context_len in lengths:
        # Need k input tokens plus 1 future token for the last label.
        if n_ids <= context_len:
            continue
        last_start = n_ids - context_len
        for start in range(0, last_start):
            input_ids = ids[start : start + context_len]
            label_ids = ids[start + 1 : start + context_len + 1]
            windows.append((input_ids, label_ids, context_len))
    return windows


# ---------------------------------------------------------------------------
# Step 4: PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------

class NextWordDataset(Dataset):
    """One teacher-forcing window per item."""

    def __init__(self, windows: list[tuple[list[int], list[int], int]]) -> None:
        """Store precomputed windows.

        Args:
            windows: Output of make_windows().
        """
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        """Return one example.

        Args:
            index: Row in self.windows.

        Returns:
            Dict with input_ids, labels, and the true length k.
        """
        input_ids, label_ids, context_len = self.windows[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "length": context_len,
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    """Pad a list of windows so they form one tensor batch.

    Shorter windows are left-padded. Real tokens sit on the right,
    which is the usual convention for RNNs.

    Args:
        batch: List of dataset items.
        pad_id: Vocab id for <pad>. Loss should ignore this id.

    Returns:
        Dict of tensors:
            input_ids: (batch, max_len)
            labels:    (batch, max_len)
            lengths:   (batch,)
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for row, item in enumerate(batch):
        context_len = item["length"]
        # Place tokens at the end of the row (left pad).
        input_ids[row, max_len - context_len :] = item["input_ids"]
        labels[row, max_len - context_len :] = item["labels"]
        lengths[row] = context_len

    return {
        "input_ids": input_ids,
        "labels": labels,
        "lengths": lengths,
    }


def make_loader(
    dataset: Dataset,
    pad_id: int,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap a NextWordDataset in a DataLoader.

    Args:
        dataset: Train, val, or test dataset.
        pad_id: Padding id from the vocab.
        batch_size: Examples per batch.
        shuffle: True only for the training loader.
        num_workers: Keep 0 on Windows.

    Returns:
        PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch=batch, pad_id=pad_id),
    )


def build_dataloaders(
    data_url: str = DATA_URL,
    data_path: Path = DATA_FILE,
    batch_size: int = 32,
    min_freq: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    context_lengths: tuple[int, ...] = CONTEXT_LENGTHS,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[dict, Vocab, dict]:
    """Full pipeline: download, tokenize, split, window, loaders.

    Args:
        data_url: Remote corpus URL.
        data_path: Local file path.
        batch_size: Loader batch size.
        min_freq: Rare-word cutoff when building vocab.
        train_ratio: Train split fraction.
        val_ratio: Val split fraction.
        context_lengths: Window sizes (4-16).
        seed: RNG seed for shuffling the train loader.
        num_workers: DataLoader workers.

    Returns:
        loaders: Dict with keys train, val, test.
        vocab: Vocabulary fitted on train tokens only.
        stats: Token counts, window counts, vocab size.
    """
    random.seed(seed)
    download_corpus(url=data_url, path=data_path)
    raw_text = read_text(path=data_path)
    tokens = tokenize(text=raw_text)

    train_tokens, val_tokens, test_tokens = split_tokens(
        tokens=tokens,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # Vocab from train only. Val/test words not seen in train become <unk>.
    vocab = Vocab(
        tokens=train_tokens,
        special_tokens=SPECIAL_TOKENS,
        min_freq=min_freq,
    )
    train_ids = vocab.encode(tokens=train_tokens)
    val_ids = vocab.encode(tokens=val_tokens)
    test_ids = vocab.encode(tokens=test_tokens)

    train_dataset = NextWordDataset(
        windows=make_windows(ids=train_ids, lengths=context_lengths)
    )
    val_dataset = NextWordDataset(
        windows=make_windows(ids=val_ids, lengths=context_lengths)
    )
    test_dataset = NextWordDataset(
        windows=make_windows(ids=test_ids, lengths=context_lengths)
    )

    pad_id = vocab.pad_id()
    loaders = {
        "train": make_loader(
            dataset=train_dataset,
            pad_id=pad_id,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": make_loader(
            dataset=val_dataset,
            pad_id=pad_id,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": make_loader(
            dataset=test_dataset,
            pad_id=pad_id,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    stats = {
        "tokens": {
            "train": len(train_tokens),
            "val": len(val_tokens),
            "test": len(test_tokens),
        },
        "windows": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "vocab_size": len(vocab),
    }
    return loaders, vocab, stats


def preview(n_examples: int = 4, batch_size: int = 4) -> None:
    """Print split sizes and a few context -> next-word examples.

    Args:
        n_examples: How many rows from the first batch to print.
        batch_size: Loader batch size used for the preview.
    """
    loaders, vocab, stats = build_dataloaders(batch_size=batch_size)
    print("corpus tokens:", stats["tokens"])
    print("windows:", stats["windows"])
    print("vocab size:", stats["vocab_size"])
    print()

    batch = next(iter(loaders["train"]))
    print("batch input_ids shape:", tuple(batch["input_ids"].shape))
    print("batch labels shape:   ", tuple(batch["labels"].shape))
    print()

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    lengths = batch["lengths"]
    n_print = min(n_examples, input_ids.size(0))

    for row in range(n_print):
        context_len = int(lengths[row])
        context_words = vocab.decode(ids=input_ids[row, -context_len:].tolist())
        label_words = vocab.decode(ids=labels[row, -context_len:].tolist())
        print("k=%2d  %s" % (context_len, " ".join(context_words)))
        print("      -> next-word target:", label_words[-1])
        print()


if __name__ == "__main__":
    preview(n_examples=4, batch_size=4)