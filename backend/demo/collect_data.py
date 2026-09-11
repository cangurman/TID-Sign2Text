"""Webcam data collection tool for TID-Sign2Text.

Records short landmark sequences for one phrase label from the webcam and
saves them as raw .npy files, ready for training - no video files needed.

Usage (from repo root):
    backend\\venv\\Scripts\\python.exe backend\\demo\\collect_data.py --label yardim_istiyorum --samples 30
Controls in the window:
    SPACE  start recording the next sample (after a 3-2-1 countdown)
    Q/ESC  quit early (already saved samples are kept)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from features import extract_frame_features  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # backend/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="class slug, e.g. yardim_istiyorum")
    parser.add_argument("--samples", type=int, default=30, help="number of samples to record")
    parser.add_argument("--frames", type=int, default=60, help="frames per sample (~2s at 30fps)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=root / "data" / "landmarks")
    args = parser.parse_args()

    out_dir = args.output / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.npy")))
    print(f"Label '{args.label}': {existing} existing sample(s), recording {args.samples} more.")

    import mediapipe as mp

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Webcam could not be opened.")

    recorded = 0
    with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
        state = "idle"  # idle -> countdown -> recording
        countdown_end = 0.0
        buffer: list[np.ndarray] = []
        while recorded < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if state == "recording":
                buffer.append(extract_frame_features(results))
                if len(buffer) >= args.frames:
                    idx = existing + recorded
                    np.save(out_dir / f"sample_{idx:03d}.npy", np.stack(buffer).astype(np.float32))
                    recorded += 1
                    buffer = []
                    state = "idle"
            elif state == "countdown" and time.time() >= countdown_end:
                state = "recording"

            # HUD
            if state == "idle":
                msg = f"[{args.label}] {recorded}/{args.samples}  SPACE=record  Q=quit"
                color = (255, 255, 255)
            elif state == "countdown":
                msg = f"Get ready... {countdown_end - time.time():.1f}"
                color = (0, 255, 255)
            else:
                msg = f"RECORDING {len(buffer)}/{args.frames}"
                color = (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                if hand:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, mp.solutions.holistic.HAND_CONNECTIONS)
            cv2.imshow("TID-Sign2Text - collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and state == "idle":
                state = "countdown"
                countdown_end = time.time() + 3.0

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {recorded} new sample(s) to {out_dir}")


if __name__ == "__main__":
    main()
