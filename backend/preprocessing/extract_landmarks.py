"""Extract MediaPipe Holistic landmarks from videos to .npy sequences.

Expected input layout:  data/raw_videos/<label>/*.mp4  (label = class slug)
Output layout:          data/landmarks/<label>/<video_stem>.npy  (T, 258) float32

Videos placed directly in data/raw_videos/ (no label folder) are skipped with
a warning. Already-extracted files are skipped unless --overwrite is given.

Usage (from repo root):
    backend\\venv\\Scripts\\python.exe backend\\preprocessing\\extract_landmarks.py
    ... extract_landmarks.py --input <dir> --output <dir> --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FRAME_DIM, extract_frame_features  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract_video(video_path: Path, holistic) -> np.ndarray:
    """Run Holistic over every frame of a video, return (T, 258) array."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        frames.append(extract_frame_features(results))
    cap.release()
    if not frames:
        return np.zeros((0, FRAME_DIM), dtype=np.float32)
    return np.stack(frames).astype(np.float32)


def extract_video_with_face(
    video_path: Path, holistic, face_indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Like extract_video, but also returns (T, len(face_indices)*3) face features."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    face_frames: list[np.ndarray] = []
    n_face = len(face_indices)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        frames.append(extract_frame_features(results))
        face = np.zeros((n_face, 3), dtype=np.float32)
        if results.face_landmarks:
            lms = results.face_landmarks.landmark
            face = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in face_indices], dtype=np.float32)
        face_frames.append(face.flatten())
    cap.release()
    if not frames:
        return np.zeros((0, FRAME_DIM), dtype=np.float32), np.zeros((0, n_face * 3), dtype=np.float32)
    return np.stack(frames).astype(np.float32), np.stack(face_frames).astype(np.float32)


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # backend/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data" / "raw_videos")
    parser.add_argument("--output", type=Path, default=root / "data" / "landmarks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--face", action="store_true",
                        help="also extract lips+eyebrow face landmarks into <output>_face/ "
                             "(TİD codes negation/questions non-manually; kept separate so the "
                             "258-dim pipeline is untouched until face is integrated)")
    args = parser.parse_args()

    import mediapipe as mp  # imported late: slow import

    face_indices: list[int] = []
    if args.face:
        # Derive lips + eyebrow indices from mediapipe's own connection sets
        # (deterministic per mediapipe version; no hardcoded index lists).
        fm = mp.solutions.face_mesh
        idx: set[int] = set()
        for conn in (fm.FACEMESH_LIPS, fm.FACEMESH_LEFT_EYEBROW, fm.FACEMESH_RIGHT_EYEBROW):
            for a, b in conn:
                idx.update((a, b))
        face_indices = sorted(idx)
        face_output = args.output.parent / (args.output.name + "_face")
        print(f"[face] {len(face_indices)} face landmarks per frame -> {face_output}")

    label_dirs = sorted(d for d in args.input.iterdir() if d.is_dir()) if args.input.exists() else []
    loose = [f for f in args.input.glob("*") if f.suffix.lower() in VIDEO_EXTENSIONS] if args.input.exists() else []
    if loose:
        print(f"[warn] {len(loose)} video(s) directly under {args.input} skipped - "
              f"put each video inside a <label>/ subfolder.")
    if not label_dirs:
        print(f"No label folders found in {args.input}. "
              f"Create e.g. {args.input / 'yardim_istiyorum'} and put MP4s inside.")
        return

    total = 0
    for label_dir in label_dirs:
        videos = sorted(f for f in label_dir.glob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
        out_dir = args.output / label_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for video in videos:
            out_file = out_dir / (video.stem + ".npy")
            if out_file.exists() and not args.overwrite:
                print(f"[skip] {out_file} exists")
                continue
            # Fresh Holistic per video: tracker state leaks across videos and
            # makes extraction depend on processing order otherwise.
            with mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                if args.face:
                    seq, face_seq = extract_video_with_face(video, holistic, face_indices)
                else:
                    seq = extract_video(video, holistic)
            if seq.shape[0] == 0:
                print(f"[warn] no frames decoded: {video}")
                continue
            detected = float(np.mean(np.any(seq != 0.0, axis=1)))
            np.save(out_file, seq)
            if args.face:
                face_dir = args.output.parent / (args.output.name + "_face") / label_dir.name
                face_dir.mkdir(parents=True, exist_ok=True)
                np.save(face_dir / out_file.name, face_seq)
            total += 1
            print(f"[ok] {label_dir.name}/{video.name}: {seq.shape[0]} frames, "
                  f"{detected:.0%} frames with detections -> {out_file.name}")
    print(f"Done. {total} file(s) written to {args.output}")


if __name__ == "__main__":
    main()
