"""Shared landmark feature utilities for TID-Sign2Text.

Single source of truth for the per-frame feature vector used by the
extractor, the trainer, the live demo and the WebSocket server.

Feature vector layout per frame (258 floats):
    pose:       33 landmarks * (x, y, z, visibility) = 132
    left hand:  21 landmarks * (x, y, z)             = 63
    right hand: 21 landmarks * (x, y, z)             = 63
"""
from __future__ import annotations

import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM = POSE_LANDMARKS * 4
HAND_DIM = HAND_LANDMARKS * 3
FRAME_DIM = POSE_DIM + 2 * HAND_DIM  # 258

# MediaPipe pose indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Left/right landmark pairs of the MediaPipe pose model, used for mirroring.
POSE_LR_PAIRS: list[tuple[int, int]] = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
    (31, 32),
]

# Bone (parent, child) pairs for the bone-vector stream (SAM-SLR style):
# bone vectors drop absolute body geometry, which helps signer independence.
POSE_BONES: list[tuple[int, int]] = [
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24),
]
HAND_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]
STREAMS_DIM = FRAME_DIM + 3 * (len(POSE_BONES) + 2 * len(HAND_BONES)) + FRAME_DIM  # 660


def extract_frame_features(results) -> np.ndarray:
    """Build the 258-float feature vector from a MediaPipe Holistic result.

    Missing components (undetected pose/hands) are zero-filled.
    """
    pose = np.zeros((POSE_LANDMARKS, 4), dtype=np.float32)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        )
    left = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        left = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
    right = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    if results.right_hand_landmarks:
        right = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )
    return np.concatenate([pose.flatten(), left.flatten(), right.flatten()])


def _split(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (T, 258) into pose (T,33,4), left (T,21,3), right (T,21,3) views."""
    t = seq.shape[0]
    pose = seq[:, :POSE_DIM].reshape(t, POSE_LANDMARKS, 4)
    left = seq[:, POSE_DIM:POSE_DIM + HAND_DIM].reshape(t, HAND_LANDMARKS, 3)
    right = seq[:, POSE_DIM + HAND_DIM:].reshape(t, HAND_LANDMARKS, 3)
    return pose, left, right


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize a raw (T, 258) sequence for model input.

    Per frame: translate x/y/z so the shoulder midpoint is the origin, then
    scale by the shoulder distance. Frames without a detected pose are left
    centered at zero. Zero-filled (missing) hands stay zero.
    """
    seq = seq.astype(np.float32, copy=True)
    pose, left, right = _split(seq)

    ls = pose[:, LEFT_SHOULDER, :3]
    rs = pose[:, RIGHT_SHOULDER, :3]
    center = (ls + rs) / 2.0                      # (T, 3)
    scale = np.linalg.norm(ls - rs, axis=1)       # (T,)
    valid = scale > 1e-4
    scale = np.where(valid, scale, 1.0)[:, None, None]
    center = np.where(valid[:, None], center, 0.0)[:, None, :]

    pose_missing = np.all(pose[:, :, :3] == 0.0, axis=(1, 2))
    for part, n_dims in ((pose, 4), (left, 3), (right, 3)):
        missing = np.all(part[:, :, :3] == 0.0, axis=(1, 2)) | pose_missing
        coords = (part[:, :, :3] - center) / scale
        part[:, :, :3] = np.where(missing[:, None, None], part[:, :, :3], coords)

    out = np.concatenate(
        [pose.reshape(len(seq), -1), left.reshape(len(seq), -1), right.reshape(len(seq), -1)],
        axis=1,
    )
    return out


def add_streams(seq: np.ndarray) -> np.ndarray:
    """Append bone-vector and motion streams to a normalized (T, 258) sequence.

    Output (T, 660): joints 258 + pose bones 24 + hand bones 120 + motion 258.
    Motion = per-frame temporal difference of the joint features (frame 0 = 0).
    """
    pose, left, right = _split(seq)
    parts = [seq]
    for coords, bones in (
        (pose[:, :, :3], POSE_BONES),
        (left, HAND_BONES),
        (right, HAND_BONES),
    ):
        vecs = np.stack([coords[:, c] - coords[:, p] for p, c in bones], axis=1)
        parts.append(vecs.reshape(len(seq), -1))
    motion = np.diff(seq, axis=0, prepend=seq[:1])
    parts.append(motion)
    return np.concatenate(parts, axis=1).astype(np.float32)


def prepare_input(seq: np.ndarray, seq_len: int, streams: bool = False) -> np.ndarray:
    """Standard raw-sequence -> model-input transform (resample+normalize[+streams])."""
    out = normalize_sequence(resample_sequence(seq, seq_len))
    if streams:
        out = add_streams(out)
    return out


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Uniformly resample a (T, D) sequence to (target_len, D) by frame picking."""
    t = seq.shape[0]
    if t == target_len:
        return seq
    if t == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim == 2 else FRAME_DIM), dtype=np.float32)
    idx = np.linspace(0, t - 1, target_len).round().astype(int)
    return seq[idx]


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Horizontally mirror a raw (T, 258) sequence (data augmentation).

    Flips x -> 1-x (MediaPipe coords are normalized to [0,1]), swaps
    left/right pose landmarks and swaps the two hand blocks.
    """
    seq = seq.astype(np.float32, copy=True)
    pose, left, right = _split(seq)

    for a, b in POSE_LR_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    left, right = right.copy(), left.copy()

    for part in (pose, left, right):
        nonzero = ~np.all(part[:, :, :3] == 0.0, axis=(1, 2))
        part[nonzero, :, 0] = 1.0 - part[nonzero, :, 0]

    return np.concatenate(
        [pose.reshape(len(seq), -1), left.reshape(len(seq), -1), right.reshape(len(seq), -1)],
        axis=1,
    )
