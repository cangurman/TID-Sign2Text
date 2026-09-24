"""Does a per-class logit bias, learned on unseen signers, reduce "hub" errors?

Hypothesis under test (from analyze_errors.py, unverified): on a signer the model
has not seen, some classes (ayni, onlar, pantolon, ogretmen, ...) attract far more
predictions than they should. A per-class bias b added to the logits could undo it.

Protocol, per signer-CV fold and seed:
  * training signers are split into "fit" signers and 2 "calibration" signers;
  * the model is trained on the fit signers only (+ _gecis built from their clips);
  * b is fitted on the calibration signers' logits (unseen by the model);
  * the test-fold signers are never used for fitting anything.
The fair comparison is the SAME model with b=0 vs with b. (Models see fewer signers
than in cv_signer_independent.py, so absolute accuracy is lower than that run.)
The _gecis bias is fixed at 0: calibration data has no transitions, a free bias would
just suppress that class.

Usage (from repo root):
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\logit_adjust_cv.py --fold-id 0 --seeds 0,1 --streams
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\logit_adjust_cv.py --aggregate
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_signer_independent import (  # noqa: E402
    TRANSITIONS, assign_folds, collect_real_samples, predict_logits, train_model)
from dataset import SampleRef  # noqa: E402
from make_transitions import TRANSITION_LABEL, stitch_transitions  # noqa: E402

OUT_DIR = BACKEND / "data" / "cv_logit"
PRIMARY_LAM = 0.1          # fixed before seeing results; other values are sensitivity checks only
SENSITIVITY_LAMS = (0.01, 1.0)
HUBS = ("ayni", "onlar", "pantolon", "ogretmen", "hep", "catal", "hayir")  # from analyze_errors.py


def run_fold(args, fold: int, seed: int, items, classes, fold_of) -> None:
    labels = classes + [TRANSITION_LABEL]
    idx = {c: i for i, c in enumerate(labels)}
    train_signers = sorted({s for _, _, s in items if fold_of[s] != fold})
    calib_signers = sorted(np.random.default_rng([seed, fold]).choice(train_signers, size=2, replace=False).tolist())
    fit_refs = [SampleRef(p, idx[c]) for p, c, s in items if fold_of[s] != fold and s not in calib_signers]
    calib = [(SampleRef(p, idx[c]), s) for p, c, s in items if s in calib_signers]
    test = [(SampleRef(p, idx[c]), s) for p, c, s in items if fold_of[s] == fold]

    trans_dir = OUT_DIR / f"tmp_fold{fold}_seed{seed}" / TRANSITION_LABEL
    if trans_dir.parent.exists():
        shutil.rmtree(trans_dir.parent)
    stitch_transitions(fit_refs, trans_dir, TRANSITIONS, seed=7)
    fit_refs += [SampleRef(f, idx[TRANSITION_LABEL]) for f in sorted(trans_dir.glob("*.npy"))]

    model, device = train_model(fit_refs, len(labels), args, seed, f"logit fold {fold} seed {seed}")
    z_cal = predict_logits(model, device, [r for r, _ in calib], args.streams)
    z_test = predict_logits(model, device, [r for r, _ in test], args.streams)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"fold{fold}_seed{seed}.npz",
             z_cal=z_cal, y_cal=np.array([r.label_idx for r, _ in calib]), s_cal=np.array([s for _, s in calib]),
             z_test=z_test, y_test=np.array([r.label_idx for r, _ in test]), s_test=np.array([s for _, s in test]),
             labels=np.array(labels), calib_signers=np.array(calib_signers))
    shutil.rmtree(trans_dir.parent, ignore_errors=True)
    acc = float(np.mean(z_test.argmax(1) == np.array([r.label_idx for r, _ in test])))
    print(f"[logit fold {fold} seed {seed}] fit signers {len(train_signers) - 2}, calibration {calib_signers}, "
          f"test n={len(test)} acc(b=0)={acc:.3f}", flush=True)


def fit_bias(z: np.ndarray, y: np.ndarray, lam: float, n_classes: int) -> np.ndarray:
    """Minimise CE(z + b) + lam * mean(b_real^2) over the real-class biases (last class fixed at 0)."""
    zt, yt = torch.from_numpy(z).float(), torch.from_numpy(y).long()
    b = torch.zeros(n_classes - 1, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(zt + F.pad(b, (0, 1)), yt) + lam * (b ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return np.concatenate([b.detach().numpy(), [0.0]])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def aggregate() -> None:
    files = sorted(OUT_DIR.glob("fold*_seed*.npz"))
    if not files:
        raise SystemExit(f"No fold*_seed*.npz under {OUT_DIR}")
    lams = (PRIMARY_LAM,) + SENSITIVITY_LAMS
    res = {lam: {"pred": [], "y": [], "sig": [], "seed": [], "conf": []} for lam in ("base",) + lams}
    per_run = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        labels = list(d["labels"])
        n = len(labels)
        seed = int(f.stem.split("seed")[1])
        row = {"run": f.stem}
        for key in ("base",) + lams:
            b = np.zeros(n) if key == "base" else fit_bias(d["z_cal"], d["y_cal"], key, n)
            p = softmax(d["z_test"] + b)
            res[key]["pred"].append(p.argmax(1))
            res[key]["conf"].append(p.max(1))
            res[key]["y"].append(d["y_test"])
            res[key]["sig"].append(d["s_test"])
            res[key]["seed"].append(np.full(len(d["y_test"]), seed))
            row[key] = float(np.mean(p.argmax(1) == d["y_test"]))
        per_run.append(row)
    cat = {k: {kk: np.concatenate(v) for kk, v in r.items()} for k, r in res.items()}
    y = cat["base"]["y"]
    hub_idx = [labels.index(h) for h in HUBS]

    print(f"{len(files)} runs. Primary lambda = {PRIMARY_LAM} (fixed in advance); other lambdas are sensitivity checks.\n")
    print("Per run accuracy on the held-out test signers (same model, b=0 vs fitted bias):")
    for r in per_run:
        print(f"  {r['run']:16s} base={r['base']:.3f}  " + "  ".join(f"lam{l}={r[l]:.3f}" for l in lams))

    def acc(key, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        return float(np.mean(cat[key]["pred"][m] == y[m]))
    print("\nPooled accuracy (all runs):")
    print(f"  base {acc('base'):.3f}")
    for lam in lams:
        print(f"  lam {lam:<5} {acc(lam):.3f}  (delta {100 * (acc(lam) - acc('base')):+.2f} points)")

    seeds = sorted(set(cat["base"]["seed"]))
    if len(seeds) > 1:
        deltas = [acc(PRIMARY_LAM, cat["base"]["seed"] == s) - acc("base", cat["base"]["seed"] == s) for s in seeds]
        print("  primary-lambda delta per seed: " + ", ".join(f"{100 * d:+.2f}" for d in deltas) + " points")

    sigs = sorted(set(cat["base"]["sig"]))
    d_sig = np.array([acc(PRIMARY_LAM, cat["base"]["sig"] == s) - acc("base", cat["base"]["sig"] == s) for s in sigs])
    n_sig = np.array([(cat["base"]["sig"] == s).sum() for s in sigs])
    rng = np.random.default_rng(0)
    boots = [float((d_sig[i] * n_sig[i]).sum() / n_sig[i].sum()) for i in (rng.integers(0, len(sigs), len(sigs)) for _ in range(5000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nPer-signer change with the primary lambda ({int((d_sig > 0).sum())} up, {int((d_sig < 0).sum())} down, "
          f"{int((d_sig == 0).sum())} unchanged of {len(sigs)} signers):")
    print("  " + "  ".join(f"{s}:{100 * d:+.1f}" for s, d in zip(sigs, d_sig)))
    print(f"  weighted mean change {100 * (d_sig * n_sig).sum() / n_sig.sum():+.2f} points, "
          f"95% CI over signers [{100 * lo:+.2f}, {100 * hi:+.2f}]")

    print("\nHub behaviour (predictions per true occurrence; 1.00 = ideal), before -> after (primary lambda):")
    for h, hi_ in zip(HUBS, hub_idx):
        occ = int((y == hi_).sum())
        pb, pa = (int((cat[k]["pred"] == hi_).sum()) for k in ("base", PRIMARY_LAM))
        prec = lambda k: (np.sum((cat[k]["pred"] == hi_) & (y == hi_)) / max(np.sum(cat[k]["pred"] == hi_), 1))
        print(f"  {h:9s} {pb / occ:.2f} -> {pa / occ:.2f}   precision {prec('base'):.3f} -> {prec(PRIMARY_LAM):.3f}")
    for k in ("base", PRIMARY_LAM):
        counts = Counter(cat[k]["pred"].tolist())
        real = [counts.get(i, 0) / max(int((y == i).sum()), 1) for i in range(len(labels) - 1)]
        print(f"  spread of predictions-per-occurrence over real classes ({'base' if k == 'base' else 'adjusted'}): "
              f"std {np.std(real):.3f}, max {max(real):.2f}")

    print("\nConfidence rejection (each run's predictions counted), base -> adjusted:")
    for th in (0.5, 0.7, 0.8):
        out = []
        for k in ("base", PRIMARY_LAM):
            m = cat[k]["conf"] >= th
            out.append(f"coverage {m.mean():.3f} acc {np.mean(cat[k]['pred'][m] == y[m]):.3f}")
        print(f"  conf>={th}: {out[0]}  ->  {out[1]}")


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
    for seed in [int(x) for x in args.seeds.split(",")]:
        run_fold(args, args.fold_id, seed, items, classes, fold_of)


if __name__ == "__main__":
    main()
