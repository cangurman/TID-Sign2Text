"""Synthesize a '_gecis' (transition) class from sign boundary windows.

Ghost-word failure mode: a sliding window straddling two signs shows the model
an unseen in-between motion, and softmax forces it onto the nearest real sign
(observed: ben+hasta emitting 'bekar'). Fix: teach the model an explicit
transition class built from stitched boundary segments - windows dominated by
a boundary then classify as '_gecis' and the emission layer stays silent.

Creates data/landmarks/_gecis/*.npy: last half of sign A + first half of
sign B for random pairs of different classes (train-split samples only, so
the validation protocol stays clean).

Usage:
    backend\\venv\\Scripts\\python.exe backend\\preprocessing\\make_transitions.py --count 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "pipeline"))

TRANSITION_LABEL = "_gecis"


def stitch_transitions(refs, out_dir: Path, count: int, seed: int = 7, segment: int | None = None) -> int:
    """Write `count` boundary-stitched transition clips built from `refs` into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    made = 0
    while made < count:
        a, b = rng.choice(len(refs), size=2, replace=False)
        ra, rb = refs[a], refs[b]
        if ra.label_idx == rb.label_idx:
            continue
        sa = np.load(ra.path).astype(np.float32)
        sb = np.load(rb.path).astype(np.float32)
        seg_a = segment or int(rng.integers(8, 16))
        seg_b = segment or int(rng.integers(8, 16))
        if sa.shape[0] < seg_a or sb.shape[0] < seg_b:
            continue
        np.save(out_dir / f"trans_{made:04d}.npy", np.concatenate([sa[-seg_a:], sb[:seg_b]], axis=0))
        made += 1
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60,
                        help="keep small: class imbalance vs 6 samples/real class biases the model")
    parser.add_argument("--segment", type=int, default=None,
                        help="frames from each side of the boundary (default: random 8-15 per side)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from dataset import discover_samples, stratified_split

    data_dir = BACKEND / "data" / "landmarks"
    out_dir = data_dir / TRANSITION_LABEL
    if out_dir.exists():
        for f in out_dir.glob("*.npy"):
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, labels = discover_samples(data_dir)
    samples = [s for s in samples if labels[s.label_idx] != TRANSITION_LABEL]
    # Only stitch train-split clips: validation clips must stay unseen.
    train_refs, _ = stratified_split(samples, val_ratio=0.2, seed=42)

    made = stitch_transitions(train_refs, out_dir, args.count, args.seed, args.segment)
    print(f"{made} transition samples -> {out_dir}")


if __name__ == "__main__":
    main()
