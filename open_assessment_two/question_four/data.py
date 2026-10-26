"""
Flickr30k loader from Hugging Face + simple word-level tokenizer.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from collections import Counter


SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_vocab(captions, min_freq: int = 2, max_size: int = 10000):
    counter = Counter()
    for cap in captions:
        counter.update(cap.lower().split())
    words = [w for w, c in counter.most_common() if c >= min_freq]
    words = words[: max_size - len(SPECIAL)]
    itos = SPECIAL + words
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def tokenize(text: str, stoi: dict, max_len: int = 32) -> list[int]:
    tokens = ["<bos>"] + text.lower().split() + ["<eos>"]
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens]
    ids = ids[:max_len]
    ids += [stoi["<pad>"]] * (max_len - len(ids))
    return ids


def _get_caption(item):
    """Extract caption from different HF mirror schemas."""
    cap = item.get("caption") or item.get("alt_text") or item.get("original_alt_text")
    if cap is None:
        raise KeyError(f"No caption field. Keys: {list(item.keys())}")
    if isinstance(cap, list):
        cap = cap[0]
    return cap


class Flickr30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        img_size: int = 64,
        max_len: int = 32,
        stoi: dict | None = None,
        dataset_name: str = "AnyModal/flickr30k",
        max_samples: int | None = None,
    ):
        self.max_len = max_len
        self.stoi = stoi
        raw = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))
        self.data = raw
        self.tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if not isinstance(image, torch.Tensor):
            if getattr(image, "mode", None) != "RGB":
                image = image.convert("RGB")
            image = self.tfm(image)
        else:
            image = image.float()
            if image.max() > 1:
                image = image / 255.0

        if self.stoi is None:
            raise RuntimeError("stoi vocab must be provided")
        caption = _get_caption(item)
        caption_ids = torch.tensor(tokenize(caption, self.stoi, self.max_len), dtype=torch.long)
        return image, caption_ids


def get_train_captions(
    dataset_name: str = "AnyModal/flickr30k",
    split: str = "train",
    max_samples: int | None = None,
):
    raw = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))
    captions = []
    for item in raw:
        captions.append(_get_caption(item))
    return captions