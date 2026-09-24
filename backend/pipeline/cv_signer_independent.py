"""Signer-grouped cross-validation: how well does the model do on people it never saw?

All real-sign samples (data/landmarks + data/landmarks_valheldout) are grouped by
signer; signers are split into K folds. For each fold the model is trained on the
other folds' signers (+ a _gecis class stitched from THOSE signers' clips only) and
evaluated on the held-out signers. Repeating with several seeds separates real
effects from training randomness. No checkpoint selection is done on held-out data
(final-epoch weights are evaluated), so numbers are not optimistically selected.

Usage (from repo root):
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\cv_signer_independent.py --fold-id 0 --seeds 0,1 --streams
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\cv_signer_independent.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "models"))
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import LandmarkDataset, SampleRef  # noqa: E402
from features import FRAME_DIM, STREAMS_DIM  # noqa: E402
from make_transitions import TRANSITION_LABEL, stitch_transitions  # noqa: E402
from model import build_model  # noqa: E402

CV_DIR = BACKEND / "data" / "cv"
TRANSITIONS = 200


def collect_real_samples() -> tuple[list[tuple[Path, str, str]], list[str]]:
    """Return ([(path, class_name, signer)], sorted real class names)."""
    items = []
    for root in (BACKEND / "data" / "landmarks", BACKEND / "data" / "landmarks_valheldout"):
        for cls_dir in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")):
            for f in sorted(cls_dir.glob("signer*.npy")):
                items.append((f, cls_dir.name, f.name.split("_")[0]))
    return items, sorted({c for _, c, _ in items})


def assign_folds(signers: list[str], k: int) -> dict[str, int]:
    ordered = sorted(signers, key=lambda s: int(s.removeprefix("signer")))
    random.Random(0).shuffle(ordered)
    return {s: i % k for i, s in enumerate(ordered)}


def run_fold(args: argparse.Namespace, fold: int, seed: int, items, classes, fold_of) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    labels = classes + [TRANSITION_LABEL]
    cls_idx = {c: i for i, c in enumerate(labels)}
    train_refs = [SampleRef(p, cls_idx[c]) for p, c, s in items if fold_of[s] != fold]
    test = [(SampleRef(p, cls_idx[c]), s) for p, c, s in items if fold_of[s] == fold]

    trans_dir = CV_DIR / f"tmp_fold{fold}" / TRANSITION_LABEL
    if trans_dir.parent.exists():
        shutil.rmtree(trans_dir.parent)
    stitch_transitions(train_refs, trans_dir, TRANSITIONS, seed=7)
    train_refs += [SampleRef(f, cls_idx[TRANSITION_LABEL]) for f in sorted(trans_dir.glob("*.npy"))]

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(
        LandmarkDataset(train_refs, seq_len=60, augment=True, streams=args.streams),
        batch_size=32, shuffle=True, num_workers=0)
    model = build_model(args.model, num_classes=len(labels),
                        input_dim=STREAMS_DIM if args.streams else FRAME_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"[fold {fold} seed {seed}] epoch {epoch}/{args.epochs}", flush=True)

    model.eval()
    test_ds = LandmarkDataset([r for r, _ in test], seq_len=60, augment=False, streams=args.streams)
    records = []
    with torch.no_grad():
        for i, (ref, signer) in enumerate(test):
            x, _ = test_ds[i]
            probs = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
            order = probs.argsort()[::-1]
            records.append({"signer": signer, "true": labels[ref.label_idx], "pred": labels[order[0]],
                            "top3": [labels[j] for j in order[:3]], "conf": round(float(probs[order[0]]), 3)})
    acc = sum(r["true"] == r["pred"] for r in records) / len(records)
    CV_DIR.mkdir(parents=True, exist_ok=True)
    out = {"fold": fold, "seed": seed, "model": args.model, "streams": args.streams, "epochs": args.epochs,
           "test_signers": sorted({r["signer"] for r in records}), "n": len(records), "acc": round(acc, 4),
           "records": records}
    (CV_DIR / f"fold{fold}_seed{seed}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(CV_DIR / f"tmp_fold{fold}", ignore_errors=True)
    print(f"[fold {fold} seed {seed}] signers {out['test_signers']} n={len(records)} acc={acc:.3f}", flush=True)


def aggregate() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CV_DIR.glob("fold*_seed*.json"))]
    if not runs:
        raise SystemExit(f"No fold*_seed*.json under {CV_DIR}")
    seeds = sorted({r["seed"] for r in runs})
    folds = sorted({r["fold"] for r in runs})
    print(f"{len(runs)} runs | folds {folds} | seeds {seeds}")

    def acc(recs, top3=False):
        return float(np.mean([(r["true"] in r["top3"]) if top3 else r["true"] == r["pred"] for r in recs]))

    print("\nPer fold x seed accuracy:")
    for f in folds:
        row = [f"seed{r['seed']}={r['acc']:.3f}" for r in runs if r["fold"] == f]
        signers = next(r["test_signers"] for r in runs if r["fold"] == f)
        print(f"  fold {f} {signers}: " + "  ".join(row))

    complete = [s for s in seeds if all(any(r["fold"] == f and r["seed"] == s for r in runs) for f in folds)]
    pooled = {s: [x for r in runs if r["seed"] == s for x in r["records"]] for s in complete}
    print("\nPooled over all held-out signers, per seed (top-1 / top-3):")
    for s in complete:
        print(f"  seed {s}: {acc(pooled[s]):.3f} / {acc(pooled[s], True):.3f}  (n={len(pooled[s])})")
    if len(complete) > 1:
        a = [acc(pooled[s]) for s in complete]
        print(f"  seed-to-seed: mean {np.mean(a):.3f}, std {np.std(a, ddof=1):.3f}")

    # Per-signer accuracy averaged over seeds; cluster bootstrap over signers for the CI.
    per_signer: dict[str, list[float]] = defaultdict(list)
    for s in complete:
        by = defaultdict(list)
        for r in pooled[s]:
            by[r["signer"]].append(r["true"] == r["pred"])
        for sg, v in by.items():
            per_signer[sg].append(float(np.mean(v)))
    counts = {sg: sum(1 for r in pooled[complete[0]] if r["signer"] == sg) for sg in per_signer}
    print("\nPer signer (mean over seeds):")
    for sg in sorted(per_signer, key=lambda x: int(x[6:])):
        print(f"  {sg:9s} n={counts[sg]:4d}  acc={np.mean(per_signer[sg]):.3f}")
    sg_acc = np.array([np.mean(per_signer[sg]) for sg in per_signer])
    sg_n = np.array([counts[sg] for sg in per_signer])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(5000):
        idx = rng.integers(0, len(sg_acc), len(sg_acc))
        boots.append(float((sg_acc[idx] * sg_n[idx]).sum() / sg_n[idx].sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nSigner-level: mean {sg_acc.mean():.3f}, std {sg_acc.std(ddof=1):.3f}, min {sg_acc.min():.3f}, "
          f"max {sg_acc.max():.3f}; pooled acc 95% CI (bootstrap over signers): [{lo:.3f}, {hi:.3f}]")

    per_class: dict[str, list[bool]] = defaultdict(list)
    for s in complete:
        for r in pooled[s]:
            per_class[r["true"]].append(r["pred"] == r["true"])
    worst = sorted(per_class.items(), key=lambda kv: np.mean(kv[1]))[:10]
    print("\nWeakest classes (recall pooled over signers and seeds):")
    for c, v in worst:
        confused = defaultdict(int)
        for s in complete:
            for r in pooled[s]:
                if r["true"] == c and r["pred"] != c:
                    confused[r["pred"]] += 1
        top = max(confused.items(), key=lambda kv: kv[1]) if confused else ("-", 0)
        print(f"  {c:14s} recall={np.mean(v):.3f}  most confused with: {top[0]} ({top[1]})")
    gecis = sum(1 for s in complete for r in pooled[s] if r["pred"] == TRANSITION_LABEL)
    print(f"\nReal-sign samples predicted as {TRANSITION_LABEL}: {gecis} of {sum(len(pooled[s]) for s in complete)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--model", choices=["bilstm", "transformer"], default="transformer")
    parser.add_argument("--streams", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.fold_id is None:
        raise SystemExit("--fold-id is required (or use --aggregate)")
    items, classes = collect_real_samples()
    fold_of = assign_folds(sorted({s for _, _, s in items}), args.folds)
    print("fold assignment:", {f: sorted(s for s, k in fold_of.items() if k == f) for f in range(args.folds)}, flush=True)
    for seed in [int(x) for x in args.seeds.split(",")]:
        run_fold(args, args.fold_id, seed, items, classes, fold_of)


if __name__ == "__main__":
    main()
