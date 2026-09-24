"""Evaluate the trained model on every staged video and write a JSON report.

Predicts from the already-extracted .npy landmark files (identical features to
training - no re-extraction), tags each sample as train/val using the exact
same stratified split (seed 42) as train.py, and writes
backend/data/eval_results.json for the demo gallery page.

Usage:
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\evaluate_videos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import discover_samples, stratified_split  # noqa: E402
from features import prepare_input  # noqa: E402
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402

VIDEO_SUFFIX = ".mp4"


def main() -> None:
    landmarks_dir = BACKEND / "data" / "landmarks"
    videos_dir = BACKEND / "data" / "raw_videos"
    out_file = BACKEND / "data" / "eval_results.json"

    predictor = StreamingPredictor(checkpoint=DEFAULT_CHECKPOINT)
    # Second predictor instance for per-video streaming subtitle timelines
    # (same engine as the live demo; reset between videos).
    stream = StreamingPredictor(
        checkpoint=DEFAULT_CHECKPOINT, window=30, stride=3, vote_len=5, threshold=0.5
    )
    samples, labels = discover_samples(landmarks_dir)
    if labels != predictor.labels:
        print("[warn] label set on disk differs from checkpoint labels - retrain before evaluating!")
    # Split over ALL samples exactly like train.py (per-class split, see dataset.py).
    train_refs, val_refs = stratified_split(samples, val_ratio=0.2, seed=42)  # must match train.py defaults
    val_paths = {s.path for s in val_refs}
    # Synthetic helper classes (e.g. _gecis) train the model but are excluded
    # from the reported metrics and the gallery.
    samples = [s for s in samples if not labels[s.label_idx].startswith("_")]

    results = []
    for s in samples:
        seq = np.load(s.path).astype(np.float32)
        x = prepare_input(seq, predictor.seq_len, streams=predictor.streams)
        with torch.no_grad():
            probs = torch.softmax(
                predictor.model(torch.from_numpy(x).unsqueeze(0).to(predictor.device)), dim=1
            )[0].cpu().numpy()
        order = probs.argsort()[::-1][:3]
        true_label = labels[s.label_idx]
        pred_label = predictor.labels[order[0]]
        video_rel = f"{true_label}/{s.path.stem}{VIDEO_SUFFIX}"
        if not (videos_dir / video_rel).exists():
            video_rel = None

        stream.reset()
        events = []
        for i in range(seq.shape[0]):
            p = stream.add_frame(seq[i])
            if p is not None:
                events.append({"t": round(i / 30.0, 3), "label": p.label,
                               "conf": round(p.confidence, 3), "stable": p.stable})
        results.append(
            {
                "video": video_rel,
                "true": true_label,
                "pred": pred_label,
                "conf": round(float(probs[order[0]]), 3),
                "top3": [[predictor.labels[i], round(float(probs[i]), 3)] for i in order],
                "split": "val" if s.path in val_paths else "train",
                "frames": int(seq.shape[0]),
                "correct": pred_label == true_label,
                "events": events,
            }
        )

    def acc(subset: list[dict]) -> float:
        return sum(r["correct"] for r in subset) / len(subset) if subset else 0.0

    val = [r for r in results if r["split"] == "val"]
    train = [r for r in results if r["split"] == "train"]
    real_labels = [l for l in labels if not l.startswith("_")]
    summary = {
        "num_classes": len(real_labels),
        "num_videos": len(results),
        "train_acc": round(acc(train), 4),
        "val_acc": round(acc(val), 4),
        "val_count": len(val),
        "labels": real_labels,
    }
    out_file.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"classes={summary['num_classes']} videos={summary['num_videos']} "
          f"train_acc={summary['train_acc']:.3f} val_acc={summary['val_acc']:.3f} (n={len(val)})")
    wrong = [r for r in results if not r["correct"]]
    for r in wrong:
        print(f"  WRONG [{r['split']}] {r['true']} -> {r['pred']} ({r['conf']:.2f})")
    print(f"Report -> {out_file}")


if __name__ == "__main__":
    main()
