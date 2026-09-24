"""Dataset utilities: load .npy landmark sequences, split, augment.

Data layout: data/landmarks/<label>/*.npy, each file (T, 258) raw landmarks.
"""
from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from features import add_streams, mirror_sequence, normalize_sequence, resample_sequence  # noqa: E402


@dataclass
class SampleRef:
    path: Path
    label_idx: int


def discover_samples(data_dir: Path) -> tuple[list[SampleRef], list[str]]:
    """Scan data_dir for <label>/*.npy files. Returns (samples, label_names)."""
    labels = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and list(d.glob("*.npy")))
    if not labels:
        raise FileNotFoundError(f"No labeled .npy data found under {data_dir}")
    samples = [
        SampleRef(path=f, label_idx=i)
        for i, label in enumerate(labels)
        for f in sorted((data_dir / label).glob("*.npy"))
    ]
    return samples, labels


def stratified_split(
    samples: list[SampleRef], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[SampleRef], list[SampleRef]]:
    # Per-class RNG keyed by class NAME: a class's split must not depend on which
    # other classes exist (a shared RNG stream shifted the split whenever _gecis
    # was present/absent, leaking val clips into the transition class).
    by_label: dict[str, list[SampleRef]] = {}
    for s in samples:
        by_label.setdefault(s.path.parent.name, []).append(s)
    train, val = [], []
    for name, group in by_label.items():
        idx = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))]).permutation(len(group))
        n_val = max(1, int(round(len(group) * val_ratio))) if len(group) > 1 else 0
        val.extend(group[i] for i in idx[:n_val])
        train.extend(group[i] for i in idx[n_val:])
    return train, val


class LandmarkDataset(Dataset):
    """Loads raw sequences, applies augmentation (train) and normalization."""

    def __init__(
        self,
        samples: list[SampleRef],
        seq_len: int = 60,
        augment: bool = False,
        noise_std: float = 0.01,
        mirror_prob: float = 0.5,
        crop_frac: float = 0.15,
        streams: bool = False,
    ) -> None:
        self.samples = samples
        self.seq_len = seq_len
        self.augment = augment
        self.noise_std = noise_std
        self.mirror_prob = mirror_prob
        self.crop_frac = crop_frac
        self.streams = streams

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        ref = self.samples[i]
        seq = np.load(ref.path).astype(np.float32)

        if self.augment:
            if np.random.rand() < self.mirror_prob:
                seq = mirror_sequence(seq)
            # random temporal crop (keeps at least 1-crop_frac of the sequence)
            t = seq.shape[0]
            if t > 10:
                max_cut = int(t * self.crop_frac)
                start = np.random.randint(0, max_cut + 1)
                end = t - np.random.randint(0, max_cut + 1)
                seq = seq[start:end]

        seq = resample_sequence(seq, self.seq_len)
        seq = normalize_sequence(seq)

        if self.augment and self.noise_std > 0:
            mask = seq != 0.0  # do not perturb missing (zero) landmarks
            seq = seq + mask * np.random.normal(0, self.noise_std, seq.shape).astype(np.float32)

        if self.streams:
            seq = add_streams(seq)

        return torch.from_numpy(seq), ref.label_idx
