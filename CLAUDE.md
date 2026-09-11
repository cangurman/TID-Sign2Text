# TID-Sign2Text Development Guidelines

## Project Context
This is a Real-Time Turkish Sign Language (TİD) Sign-to-Text translation system.
It converts live camera streams into Turkish sentences using MediaPipe landmark
extraction and sequence models (Bi-LSTM / Transformer).

MVP scope: **isolated phrase recognition** over 30-50 domain-specific phrases
(Emergency / Hospital front-desk scenario), not open-vocabulary continuous
translation. See `docs/PHRASES.md` for the phrase list.

## Tech Stack
- **Backend:** Python 3.12 (venv at `backend/venv`), PyTorch, MediaPipe, OpenCV, FastAPI, Uvicorn
- **Web demo:** plain HTML/JS with MediaPipe Holistic (CDN) + WebSocket
- **Mobile (later):** Flutter (Dart), camera plugin, WebSockets
- **Environment:** Windows PowerShell

## Fixed Technical Decisions
1. AI processing uses **landmark coordinates, never raw video frames**.
2. Feature vector per frame is **258 floats**: pose 33×(x,y,z,visibility) +
   left hand 21×(x,y,z) + right hand 21×(x,y,z). Defined once in
   `backend/preprocessing/features.py` — every component (extractor, trainer,
   live demo, WebSocket server, web client) must use this exact layout.
3. `.npy` files store **raw** (unnormalized) landmark sequences of shape
   `(num_frames, 258)`. Normalization (shoulder-centering + shoulder-width
   scaling) is applied at train/inference time via `normalize_sequence()`.
4. Data layout: `backend/data/raw_videos/<label>/*.mp4` and
   `backend/data/landmarks/<label>/*.npy` — the folder name is the class label
   (ASCII slug, e.g. `yardim_istiyorum`).
5. Model input is a fixed-length sequence (default 60 frames), obtained by
   uniform temporal resampling.

## Coding Principles
1. Keep modular structure: separate data processing, model definitions, and API routes.
2. Write clean, type-hinted Python and async FastAPI endpoints.
3. Ensure low-latency processing (<200 ms target per prediction).
4. OpenCV `putText` cannot render Turkish characters — use the PIL helper in
   `backend/demo/live_demo.py` for on-screen Turkish text.

## Data Source (current)
Real TİD videos come from the public AUTSL mirror on Hugging Face
(`aipieces/AUTSL`, 57 train shards of ~300 MB). Shard 1 is extracted at
`backend/data/autsl/shard_001/` (500 videos, signer0 only, `*_color.mp4` +
`*_depth.mp4`; we use color only). Label CSVs live in `backend/data/autsl/`.
`preprocessing/prepare_autsl.py --top N` stages the N most-sampled classes
into `data/raw_videos/<label>/`.

KNOWN LIMITATION (be honest about it): all current data is ONE signer
(signer0). Reported val accuracy is same-signer accuracy; signer-independent
accuracy will be much lower until multi-signer shards are added
(AUTSL baseline: 95.9% random-split vs 62.0% signer-independent).

## Common Commands (PowerShell, from repo root)
```powershell
$py = ".\backend\venv\Scripts\python.exe"
& $py backend\preprocessing\prepare_autsl.py --shard-dir backend\data\autsl\shard_001 --top 31
& $py backend\preprocessing\extract_landmarks.py          # videos -> .npy
& $py backend\demo\collect_data.py --label X --samples 30 # webcam data collection
& $py backend\pipeline\train.py                           # train + save checkpoint (--model transformer)
#   --streams: bone-vector + motion streams (input 258 -> 660; SAM-SLR recipe,
#   helps signer independence; stored in checkpoint, applied automatically at inference)
& $py backend\preprocessing\make_transitions.py           # regenerate _gecis class (after data changes)
& $py backend\pipeline\eval_signer_independent.py         # honest unseen-signer accuracy (signer10)
# backend\pipeline\run_multisigner_chain.ps1: full A/B ablation chain (background)
& $py backend\pipeline\evaluate_videos.py                 # per-video JSON report for /demo page
& $py backend\pipeline\build_subtitles.py --count 8       # streaming subtitle timelines for /watch page
& $py backend\pipeline\make_sentences.py                  # stitched sentence videos + decoding for /sentence
& $py backend\demo\predict_video.py <video|folder>        # offline video -> text
& $py backend\demo\live_demo.py                           # real-time webcam demo
& $py -m uvicorn main:app --app-dir backend --port 8080   # /demo gallery, /watch subtitles, /sentence continuous
```
Pipeline rule: always create a FRESH mediapipe Holistic instance per video -
tracker state leaks across videos and makes extraction order-dependent.
Note: port 8000 is occupied by an unrelated Python 3.14 process on this
machine - use 8080. Browsers cannot play the raw AUTSL FMP4 videos; serve the
H.264 re-encodes in `data/web_videos/` (regenerate with the bundled
imageio-ffmpeg binary if new videos are staged).

## Core Principles (user-mandated, non-negotiable)
1. **Errors trend to zero.** Recognition/translation errors (ghost words,
   swallowed words, weak classes) are never "acceptable" — each must be traced
   to a root cause, attached to a fix in the roadmap, and re-measured every
   iteration. Show errors honestly; never hide them, never accept them.
2. **No fabricated facts or interpretations.** Every claim must be measured
   (test/metric) or sourced (literature/URL); anything uncertain is labeled
   unverified. Guesses must be declared as guesses.

## Evaluation Discipline
- `train.py` and `evaluate_videos.py` MUST use the same split (seed 42,
  val_ratio 0.2, stratified). If you change one, change the other.
- After every retrain, rerun `evaluate_videos.py` so `/demo` shows current
  numbers. Never report train accuracy as "accuracy".
