"""Continuous sign-to-text: build sentence videos and decode them word by word.

Concatenates validation-split word clips (never seen in training) into
sentence videos, then runs the streaming predictor over the stitched video's
landmarks with a word-emission layer (stable + repeated + changed = emit).
The emitted gloss sequence is mapped to a fluent Turkish sentence via the
domain-specific template table - the "Sign2Text" step of the MVP spec.

Outputs:
    backend/data/sentences/<name>.mp4        (H.264, browser-playable)
    backend/data/web_videos/sentences/...    (served copy)
    backend/data/sentences.json              (timeline for the /sentence page)

Usage:
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\make_sentences.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import extract_video  # noqa: E402
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402

# gloss sequence -> fluent Turkish (domain-specific template layer)
SENTENCES: list[dict] = [
    {"name": "ben_acikmak", "glosses": ["ben", "acikmak"], "turkish": "Ben acıktım."},
    {"name": "eczane_uzak", "glosses": ["eczane", "uzak"], "turkish": "Eczane uzak."},
    {"name": "tehlike_kacmak", "glosses": ["tehlike", "kacmak"], "turkish": "Tehlike var, kaçın!"},
    {"name": "tehlike_polis_kacmak", "glosses": ["tehlike", "polis", "kacmak"],
     "turkish": "Tehlike var! Polis çağırın, kaçın!"},
    {"name": "ben_hasta", "glosses": ["ben", "hasta"], "turkish": "Ben hastayım."},
    {"name": "hastane_uzak", "glosses": ["hastane", "uzak"], "turkish": "Hastane uzak."},
    {"name": "ilac_yok", "glosses": ["ilac", "yok"], "turkish": "İlaç yok."},
    {"name": "doktor_beklemek", "glosses": ["doktor", "beklemek"], "turkish": "Doktoru bekliyorum."},
]
TEMPLATES = {tuple(s["glosses"]): s["turkish"] for s in SENTENCES}

WINDOW = 45  # 30-frame windows lose too much context on transition-heavy stitched video
STRIDE = 3
VOTE_LEN = 5
THRESHOLD = 0.60
EMIT_REPEAT = 3  # stable ticks with the same label required before emitting

FPS = 30.0


def val_video_for(label: str) -> Path:
    """Best validation-split video for a class: correct first, then highest confidence.

    Uses eval_results.json (run evaluate_videos.py after every retrain) so the
    sentence demo is stitched from each word's most reliable held-out clip.
    """
    eval_file = BACKEND / "data" / "eval_results.json"
    rows = json.loads(eval_file.read_text(encoding="utf-8"))["results"]
    candidates = [r for r in rows if r["split"] == "val" and r["true"] == label and r["video"]]
    if not candidates:
        raise LookupError(f"No validation video for class {label!r}")
    best = sorted(candidates, key=lambda r: (not r["correct"], -r["conf"]))[0]
    return BACKEND / "data" / "raw_videos" / best["video"]


def concat_videos(inputs: list[Path], out: Path) -> None:
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    streams = "".join(f"[{i}:v]" for i in range(len(inputs)))
    cmd += [
        "-filter_complex", f"{streams}concat=n={len(inputs)}:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)


def decode(seq: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Streaming decode: returns (per-tick events, emitted words)."""
    predictor = StreamingPredictor(
        checkpoint=DEFAULT_CHECKPOINT,
        window=WINDOW, stride=STRIDE, vote_len=VOTE_LEN, threshold=THRESHOLD,
    )
    events, emissions = [], []
    last_emitted, pending, pending_n = None, None, 0
    for i in range(seq.shape[0]):
        pred = predictor.add_frame(seq[i])
        if pred is None:
            continue
        t = round(i / FPS, 3)
        events.append({"t": t, "label": pred.label, "conf": round(pred.confidence, 3),
                       "stable": pred.stable})
        if pred.stable:
            if pred.label.startswith("_"):  # transition class: stay silent (keep candidate)
                continue
            if pred.label == pending:
                pending_n += 1
            else:
                pending, pending_n = pred.label, 1
            if pending_n >= EMIT_REPEAT and pred.label != last_emitted:
                emissions.append({"t": t, "word": pred.label})
                last_emitted = pred.label
    return events, emissions


def main() -> None:
    out_dir = BACKEND / "data" / "sentences"
    web_dir = BACKEND / "data" / "web_videos" / "sentences"
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)

    import mediapipe as mp

    report = []
    for spec in SENTENCES:
        clips = [val_video_for(g) for g in spec["glosses"]]
        mp4 = out_dir / f"{spec['name']}.mp4"
        concat_videos(clips, mp4)
        shutil.copy2(mp4, web_dir / mp4.name)

        # Fresh Holistic per video: tracker state leaks across videos otherwise,
        # making results depend on processing order (non-deterministic pipeline).
        with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
            seq = extract_video(mp4, holistic)
            events, emissions = decode(seq)
            emitted = [e["word"] for e in emissions]
            fluent = TEMPLATES.get(tuple(emitted))
            report.append(
                {
                    "video": f"sentences/{mp4.name}",
                    "name": spec["name"],
                    "glosses_true": spec["glosses"],
                    "turkish_true": spec["turkish"],
                    "words_emitted": emitted,
                    "turkish_pred": fluent if fluent else " ".join(emitted).upper(),
                    "exact_match": emitted == spec["glosses"],
                    "duration": round(seq.shape[0] / FPS, 2),
                    "events": events,
                    "emissions": emissions,
                }
            )
            status = "OK" if emitted == spec["glosses"] else "MISMATCH"
            print(f"[{status}] {spec['name']}: true={spec['glosses']} emitted={emitted}")

    out_file = BACKEND / "data" / "sentences.json"
    out_file.write_text(json.dumps({"sentences": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
