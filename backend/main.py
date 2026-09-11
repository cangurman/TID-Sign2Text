"""TID-Sign2Text FastAPI server.

Endpoints:
    GET  /            -> serves the live webcam client (web_client/index.html)
    GET  /demo        -> video gallery: staged videos + model predictions
    GET  /api/results -> JSON produced by pipeline/evaluate_videos.py
    GET  /videos/...  -> static serving of data/raw_videos
    GET  /health      -> model/checkpoint status
    WS   /ws/stream   -> real-time landmark frames in, predictions out

WebSocket protocol (JSON):
    client -> server:  {"frame": [258 floats]}      one frame of raw landmarks
                       {"cmd": "reset"}             clear the sliding window
    server -> client:  {"text": str, "confidence": float, "stable": bool}
                       {"error": str}

Run (from repo root):
    backend\\venv\\Scripts\\python.exe -m uvicorn main:app --app-dir backend --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND / "pipeline"))
from predictor import DEFAULT_CHECKPOINT, StreamingPredictor  # noqa: E402

WEB_CLIENT = BACKEND.parent / "web_client" / "index.html"
GALLERY = BACKEND.parent / "web_client" / "gallery.html"
WATCH = BACKEND.parent / "web_client" / "watch.html"
RAW_VIDEOS = BACKEND / "data" / "raw_videos"
WEB_VIDEOS = BACKEND / "data" / "web_videos"  # H.264 re-encodes (raw AUTSL is FMP4, not browser-playable)
EVAL_RESULTS = BACKEND / "data" / "eval_results.json"
SUBTITLES = BACKEND / "data" / "subtitles.json"

app = FastAPI(title="TID-Sign2Text API")

_video_dir = WEB_VIDEOS if WEB_VIDEOS.exists() else RAW_VIDEOS
if _video_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/videos", StaticFiles(directory=_video_dir), name="videos")


@app.middleware("http")
async def no_cache_videos(request, call_next):
    # The first deploy served non-browser-playable FMP4 files; force browsers
    # to revalidate instead of replaying their cached broken copies.
    response = await call_next(request)
    if request.url.path.startswith("/videos/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/watch", response_model=None)
async def watch() -> FileResponse | JSONResponse:
    if WATCH.exists():
        return FileResponse(WATCH)
    return JSONResponse({"error": "web_client/watch.html not found"}, status_code=404)


@app.get("/sentence", response_model=None)
async def sentence_page() -> FileResponse | JSONResponse:
    page = BACKEND.parent / "web_client" / "sentence.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"error": "web_client/sentence.html not found"}, status_code=404)


@app.get("/api/sentences")
async def api_sentences() -> JSONResponse:
    f = BACKEND / "data" / "sentences.json"
    if f.exists():
        import json

        return JSONResponse(json.loads(f.read_text(encoding="utf-8")))
    return JSONResponse(
        {"error": "sentences.json yok - once pipeline/make_sentences.py calistirin"},
        status_code=404,
    )


@app.get("/api/subtitles")
async def api_subtitles() -> JSONResponse:
    if SUBTITLES.exists():
        import json

        return JSONResponse(json.loads(SUBTITLES.read_text(encoding="utf-8")))
    return JSONResponse(
        {"error": "subtitles.json yok - once pipeline/build_subtitles.py calistirin"},
        status_code=404,
    )


@app.get("/demo", response_model=None)
async def demo() -> FileResponse | JSONResponse:
    if GALLERY.exists():
        return FileResponse(GALLERY)
    return JSONResponse({"error": "web_client/gallery.html not found"}, status_code=404)


@app.get("/api/results")
async def api_results() -> JSONResponse:
    if EVAL_RESULTS.exists():
        import json

        return JSONResponse(json.loads(EVAL_RESULTS.read_text(encoding="utf-8")))
    return JSONResponse(
        {"error": "eval_results.json yok - once pipeline/evaluate_videos.py calistirin"},
        status_code=404,
    )


def load_predictor() -> StreamingPredictor | None:
    if DEFAULT_CHECKPOINT.exists():
        return StreamingPredictor(checkpoint=DEFAULT_CHECKPOINT)
    return None


@app.get("/", response_model=None)
async def index() -> FileResponse | JSONResponse:
    if WEB_CLIENT.exists():
        return FileResponse(WEB_CLIENT)
    return JSONResponse({"status": "ok", "hint": "web_client/index.html not found"})


@app.get("/health")
async def health() -> JSONResponse:
    ready = DEFAULT_CHECKPOINT.exists()
    labels: list[str] = []
    if ready:
        import torch

        ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
        labels = ckpt["labels"]
    return JSONResponse({"model_ready": ready, "num_classes": len(labels), "labels": labels})


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    predictor = load_predictor()  # per-connection: own sliding window
    if predictor is None:
        await ws.send_json({"error": "Model not trained yet (models/checkpoints/best.pt missing)."})
        await ws.close()
        return
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("cmd") == "reset":
                predictor.reset()
                continue
            frame = msg.get("frame")
            if frame is None:
                await ws.send_json({"error": "expected {'frame': [258 floats]}"})
                continue
            arr = np.asarray(frame, dtype=np.float32)
            try:
                pred = predictor.add_frame(arr)
            except ValueError as exc:
                await ws.send_json({"error": str(exc)})
                continue
            if pred is not None:
                await ws.send_json(
                    {
                        "text": pred.label.replace("_", " "),
                        "confidence": round(pred.confidence, 3),
                        "stable": pred.stable,
                    }
                )
    except WebSocketDisconnect:
        pass
