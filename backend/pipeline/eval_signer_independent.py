"""Signer-independent evaluation: test the model on a signer it never saw.

Training data is signer0 only; this evaluates full-sequence classification on
landmarks extracted from another signer's videos (default: signer10 from AUTSL
shard 3, staged in data/landmarks_signer10). This is the honest
generalization number - expect it to be much lower than same-signer val.

Usage:
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\eval_signer_independent.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import prepare_input  # noqa: E402
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BACKEND / "data" / "landmarks_signer10")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    predictor = StreamingPredictor(checkpoint=args.checkpoint)
    known = set(predictor.labels)

    total, correct, top3_hits = 0, 0, 0
    per_class: dict[str, list[bool]] = defaultdict(list)
    skipped = 0
    for label_dir in sorted(d for d in args.data_dir.iterdir() if d.is_dir()):
        if label_dir.name not in known:
            skipped += len(list(label_dir.glob("*.npy")))
            continue
        for f in sorted(label_dir.glob("*.npy")):
            seq = np.load(f).astype(np.float32)
            x = prepare_input(seq, predictor.seq_len, streams=predictor.streams)
            with torch.no_grad():
                probs = torch.softmax(
                    predictor.model(torch.from_numpy(x).unsqueeze(0).to(predictor.device)), dim=1
                )[0].cpu().numpy()
            order = probs.argsort()[::-1]
            pred = predictor.labels[order[0]]
            ok = pred == label_dir.name
            total += 1
            correct += ok
            top3_hits += label_dir.name in [predictor.labels[i] for i in order[:3]]
            per_class[label_dir.name].append(ok)
            if not ok:
                print(f"  wrong: {label_dir.name} -> {pred} ({probs[order[0]]:.2f})")

    if not total:
        raise SystemExit(f"No evaluable samples in {args.data_dir}")
    print(f"\nSIGNER-INDEPENDENT accuracy: {correct}/{total} = {correct / total:.3f} "
          f"(top-3: {top3_hits / total:.3f}); {skipped} sample(s) of unknown classes skipped")
    hard = [c for c, oks in per_class.items() if not any(oks)]
    if hard:
        print(f"classes fully missed: {', '.join(sorted(hard))}")


if __name__ == "__main__":
    main()
