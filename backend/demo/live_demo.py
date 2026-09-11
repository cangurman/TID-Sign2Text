"""Real-time webcam demo: MediaPipe Holistic -> StreamingPredictor -> subtitle.

Usage (from repo root, after training):
    backend\\venv\\Scripts\\python.exe backend\\demo\\live_demo.py
Controls: Q/ESC quit, R reset the prediction window.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(BACKEND / "pipeline"))
from features import extract_frame_features  # noqa: E402
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402


def make_text_drawer():
    """OpenCV putText cannot render Turkish characters; use PIL if available."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_big = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
        font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)

        def draw(frame: np.ndarray, big: str, small: str, color: tuple[int, int, int]) -> np.ndarray:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(img)
            h = img.height
            d.rectangle([(0, h - 90), (img.width, h)], fill=(0, 0, 0))
            d.text((15, h - 82), big, font=font_big, fill=color)
            d.text((15, h - 32), small, font=font_small, fill=(200, 200, 200))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return draw
    except Exception:
        def draw(frame: np.ndarray, big: str, small: str, color: tuple[int, int, int]) -> np.ndarray:
            h = frame.shape[0]
            cv2.rectangle(frame, (0, h - 90), (frame.shape[1], h), (0, 0, 0), -1)
            cv2.putText(frame, big, (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color[::-1], 2)
            cv2.putText(frame, small, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return frame

        return draw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}\nTrain first: python backend/pipeline/train.py")

    predictor = StreamingPredictor(checkpoint=args.checkpoint, threshold=args.threshold)
    print(f"Model loaded: {len(predictor.labels)} classes, seq_len={predictor.seq_len}, device={predictor.device}")

    import mediapipe as mp

    draw_text = make_text_drawer()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Webcam could not be opened.")

    subtitle, sub_conf = "", 0.0
    fps, t_prev = 0.0, time.time()
    with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            pred = predictor.add_frame(extract_frame_features(results))
            if pred is not None and pred.stable:
                subtitle, sub_conf = pred.label.replace("_", " "), pred.confidence

            t_now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                if hand:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand, mp.solutions.holistic.HAND_CONNECTIONS)

            fill = min(len(predictor.frames), predictor.window)
            status = f"fps {fps:4.1f} | window {fill}/{predictor.window} | conf {sub_conf:.2f} | R=reset Q=quit"
            color = (0, 255, 128) if subtitle else (255, 255, 255)
            frame = draw_text(frame, subtitle or "...", status, color)
            cv2.imshow("TID-Sign2Text - live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                predictor.reset()
                subtitle, sub_conf = "", 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
