"""Offline sign-to-text: predict the phrase in a video file (or folder of videos).

Usage (from repo root, after training):
    backend\\venv\\Scripts\\python.exe backend\\demo\\predict_video.py <video.mp4 | folder> [--top 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(BACKEND / "pipeline"))
from extract_landmarks import VIDEO_EXTENSIONS, extract_video  # noqa: E402
from features import prepare_input  # noqa: E402
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402


def predict_sequence(predictor: StreamingPredictor, seq: np.ndarray, top: int) -> list[tuple[str, float]]:
    seq = prepare_input(seq, predictor.seq_len, streams=predictor.streams)
    with torch.no_grad():
        x = torch.from_numpy(seq).unsqueeze(0).to(predictor.device)
        probs = torch.softmax(predictor.model(x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1][:top]
    return [(predictor.labels[i], float(probs[i])) for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="video file or folder of videos")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint} - train first.")
    videos = (
        [args.target]
        if args.target.is_file()
        else sorted(f for f in args.target.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
    )
    if not videos:
        raise SystemExit(f"No videos found at {args.target}")

    predictor = StreamingPredictor(checkpoint=args.checkpoint)
    import mediapipe as mp

    with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
        for video in videos:
            seq = extract_video(video, holistic)
            if seq.shape[0] == 0:
                print(f"{video.name}: no frames decoded")
                continue
            ranking = predict_sequence(predictor, seq, args.top)
            best, conf = ranking[0]
            rest = "  ".join(f"{l}={p:.2f}" for l, p in ranking[1:])
            print(f"{video.name} ({seq.shape[0]} kare) -> {best.replace('_', ' ').upper()} "
                  f"(güven {conf:.2f})   [{rest}]")


if __name__ == "__main__":
    main()
