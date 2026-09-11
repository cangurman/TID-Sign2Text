"""Build subtitle timelines by replaying the real-time predictor over videos.

Runs the exact same StreamingPredictor used by the live webcam demo and the
WebSocket server frame-by-frame over each video's extracted landmarks, and
records every prediction with its timestamp. The /watch page plays the video
and shows these predictions as live subtitles - i.e. an honest offline replay
of the real-time pipeline (precomputed because the browser cannot run the
PyTorch model).

Usage:
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\build_subtitles.py --count 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402

# Short clips (~1.5-2.5 s): a 1-second window with a small stride gives the
# subtitle a real-time feel instead of one prediction at the very end.
WINDOW = 30
STRIDE = 3
VOTE_LEN = 5
THRESHOLD = 0.5


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps if fps > 1 else 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8, help="videos to include")
    parser.add_argument("--eval-file", type=Path, default=BACKEND / "data" / "eval_results.json")
    parser.add_argument("--out", type=Path, default=BACKEND / "data" / "subtitles.json")
    args = parser.parse_args()

    eval_data = json.loads(args.eval_file.read_text(encoding="utf-8"))
    # Prefer validation videos (never seen in training), one per class, correct first.
    candidates = [r for r in eval_data["results"] if r["video"]]
    candidates.sort(key=lambda r: (r["split"] != "val", not r["correct"], -r["conf"]))
    picked, seen_classes = [], set()
    for r in candidates:
        if r["true"] in seen_classes:
            continue
        picked.append(r)
        seen_classes.add(r["true"])
        if len(picked) >= args.count:
            break

    entries = []
    for r in picked:
        npy = BACKEND / "data" / "landmarks" / r["true"] / (Path(r["video"]).stem + ".npy")
        mp4 = BACKEND / "data" / "raw_videos" / r["video"]
        seq = np.load(npy).astype(np.float32)
        fps = video_fps(mp4)

        predictor = StreamingPredictor(
            checkpoint=DEFAULT_CHECKPOINT,
            window=WINDOW, stride=STRIDE, vote_len=VOTE_LEN, threshold=THRESHOLD,
        )
        events = []
        for i in range(seq.shape[0]):
            pred = predictor.add_frame(seq[i])
            if pred is not None:
                events.append(
                    {
                        "t": round(i / fps, 3),
                        "label": pred.label,
                        "conf": round(pred.confidence, 3),
                        "stable": pred.stable,
                    }
                )
        entries.append(
            {
                "video": r["video"],
                "true": r["true"],
                "split": r["split"],
                "fps": fps,
                "duration": round(seq.shape[0] / fps, 2),
                "final_pred": r["pred"],
                "final_correct": r["correct"],
                "events": events,
            }
        )
        stable_labels = [e["label"] for e in events if e["stable"]]
        outcome = stable_labels[-1] if stable_labels else "(no stable prediction)"
        print(f"{r['video']}: {len(events)} events, last stable = {outcome} (true: {r['true']})")

    args.out.write_text(
        json.dumps({"window": WINDOW, "stride": STRIDE, "videos": entries}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
