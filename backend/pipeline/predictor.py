"""Shared real-time inference logic: sliding window + smoothing.

Used by both the OpenCV live demo and the FastAPI WebSocket server so the
prediction behavior is identical everywhere.
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "models"))
sys.path.insert(0, str(BACKEND / "preprocessing"))
from features import FRAME_DIM, STREAMS_DIM, prepare_input  # noqa: E402
from model import build_model  # noqa: E402

DEFAULT_CHECKPOINT = BACKEND / "models" / "checkpoints" / "best.pt"


@dataclass
class Prediction:
    label: str
    confidence: float
    stable: bool  # True once the same label won the majority vote


class StreamingPredictor:
    """Feed frames one by one; get smoothed predictions back.

    - keeps a sliding window of the last `window` raw frames
    - runs the model every `stride` frames once the window is full
    - majority-votes over the last `vote_len` predictions and only reports
      a label as `stable` when it wins the vote with confidence >= threshold
    """

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        window: int = 60,
        stride: int = 5,
        vote_len: int = 8,
        threshold: float = 0.6,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.labels: list[str] = ckpt["labels"]
        self.seq_len: int = ckpt["seq_len"]
        self.streams: bool = ckpt.get("streams", False)
        input_dim = STREAMS_DIM if self.streams else FRAME_DIM
        self.model = build_model(ckpt["model"], num_classes=len(self.labels), input_dim=input_dim)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        self.window = window
        self.stride = stride
        self.threshold = threshold
        self.frames: deque[np.ndarray] = deque(maxlen=window)
        self.votes: deque[int] = deque(maxlen=vote_len)
        self._since_last = 0

    def reset(self) -> None:
        self.frames.clear()
        self.votes.clear()
        self._since_last = 0

    def add_frame(self, frame: np.ndarray) -> Prediction | None:
        """Add one raw 258-float frame. Returns a Prediction on inference ticks."""
        if frame.shape != (FRAME_DIM,):
            raise ValueError(f"Expected frame shape ({FRAME_DIM},), got {frame.shape}")
        self.frames.append(frame.astype(np.float32))
        self._since_last += 1
        if len(self.frames) < self.window or self._since_last < self.stride:
            return None
        self._since_last = 0
        return self._predict()

    def _predict(self) -> Prediction:
        seq = prepare_input(np.stack(self.frames), self.seq_len, streams=self.streams)
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        conf = float(probs[idx])

        if conf >= self.threshold:
            self.votes.append(idx)
        winner, stable = idx, False
        if self.votes:
            winner, count = Counter(self.votes).most_common(1)[0]
            stable = count >= max(2, len(self.votes) // 2 + 1) and conf >= self.threshold
        return Prediction(label=self.labels[winner], confidence=conf, stable=stable)
