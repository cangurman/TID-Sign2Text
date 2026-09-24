"""Root-cause analysis of signer-independent errors, from the CV predictions.

Reads backend/data/cv/fold*_seed*.json (written by cv_signer_independent.py), maps each
prediction back to its landmark file and measures candidate error causes:
  A. hand-detection quality vs accuracy      B. per-signer recording statistics vs accuracy
  C. are confused class pairs close in landmark space?   D. confidence: can errors be rejected?
  E. "sink" classes that attract wrong predictions
Everything printed is a measurement; interpretations are marked as hypotheses.

Usage (from repo root):
    backend\\venv\\Scripts\\python.exe backend\\pipeline\\analyze_errors.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_signer_independent import CV_DIR, assign_folds, collect_real_samples  # noqa: E402
from features import HAND_DIM, POSE_DIM, prepare_input  # noqa: E402

RIGHT_WRIST, LEFT_WRIST = 16, 15  # MediaPipe pose indices


def sample_stats(path: Path) -> dict:
    seq = np.load(path).astype(np.float32)
    t = seq.shape[0]
    pose = seq[:, :POSE_DIM].reshape(t, 33, 4)
    left = seq[:, POSE_DIM:POSE_DIM + HAND_DIM]
    right = seq[:, POSE_DIM + HAND_DIM:]
    l_ok, r_ok = np.any(left != 0, axis=1), np.any(right != 0, axis=1)
    p_ok = np.any(pose[:, :, :3] != 0, axis=(1, 2))
    sw = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    sw_mean = float(sw[sw > 1e-4].mean()) if np.any(sw > 1e-4) else 0.0
    path_len = {}
    for name, idx in (("L", LEFT_WRIST), ("R", RIGHT_WRIST)):
        xy = pose[:, idx, :2]
        path_len[name] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() / max(sw_mean, 1e-4))
    return {"T": t, "hand": float(np.mean(l_ok | r_ok)), "left": float(l_ok.mean()), "right": float(r_ok.mean()),
            "pose": float(p_ok.mean()), "shoulder_w": sw_mean,
            "left_dominant": path_len["L"] > path_len["R"]}


def main() -> None:
    items, classes = collect_real_samples()
    fold_of = assign_folds(sorted({s for _, _, s in items}), 4)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CV_DIR.glob("fold*_seed*.json"))]
    seeds = sorted({r["seed"] for r in runs})

    # records -> file paths (same deterministic order as run_fold), validated per record
    rows: dict[tuple[int, int], dict] = {}
    for r in runs:
        fold_items = [(p, c, s) for p, c, s in items if fold_of[s] == r["fold"]]
        assert len(fold_items) == len(r["records"]), "record/item count mismatch"
        for i, ((p, c, s), rec) in enumerate(zip(fold_items, r["records"])):
            assert rec["signer"] == s and rec["true"] == c, "record/item identity mismatch"
            row = rows.setdefault((r["fold"], i), {"path": p, "true": c, "signer": s, "ok": [], "conf": [], "pred": []})
            row["ok"].append(rec["pred"] == c)
            row["conf"].append(rec["conf"])
            row["pred"].append(rec["pred"])
    for row in rows.values():
        row.update(sample_stats(row["path"]))
        row["acc"] = float(np.mean(row["ok"]))
    R = list(rows.values())
    print(f"{len(R)} samples, {len(seeds)} seed(s); mean accuracy {np.mean([x['acc'] for x in R]):.3f}\n")

    print("A. Hand-detection rate (fraction of frames with any hand landmark) vs accuracy")
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)):
        sub = [x for x in R if lo <= x["hand"] < hi]
        if sub:
            print(f"   {lo:.2f}-{min(hi, 1):.2f}: n={len(sub):5d}  acc={np.mean([x['acc'] for x in sub]):.3f}")
    rho, p = spearmanr([x["hand"] for x in R], [x["acc"] for x in R])
    print(f"   per-sample Spearman(hand rate, correct): rho={rho:.3f} (p={p:.1e})")
    rho, p = spearmanr([x["T"] for x in R], [x["acc"] for x in R])
    print(f"   per-sample Spearman(frames, correct):    rho={rho:.3f} (p={p:.1e})\n")

    print("B. Per-signer recording statistics vs accuracy")
    sig = defaultdict(list)
    for x in R:
        sig[x["signer"]].append(x)
    tab = {}
    for s, v in sig.items():
        tab[s] = {"acc": np.mean([x["acc"] for x in v]), "T": np.mean([x["T"] for x in v]),
                  "hand": np.mean([x["hand"] for x in v]), "lowhand": np.mean([x["hand"] < .8 for x in v]),
                  "sw": np.mean([x["shoulder_w"] for x in v]), "ldom": np.mean([x["left_dominant"] for x in v])}
    print("   signer      n   acc  frames  hand%  <0.8hand%  shoulderW  leftdom%")
    for s in sorted(tab, key=lambda z: tab[z]["acc"]):
        t = tab[s]
        print(f"   {s:9s} {len(sig[s]):4d}  {t['acc']:.3f}  {t['T']:6.1f}  {t['hand']:.3f}  {t['lowhand']:.3f}"
              f"      {t['sw']:.3f}     {t['ldom']:.2f}")
    print("   accuracy split by hand-detection quality within each signer (low = hand rate < 0.8):")
    for s in sorted(tab, key=lambda z: tab[z]["acc"]):
        lo = [x["acc"] for x in sig[s] if x["hand"] < .8]
        hi = [x["acc"] for x in sig[s] if x["hand"] >= .8]
        if lo:
            print(f"     {s:9s} low n={len(lo):3d} acc={np.mean(lo):.3f} | high n={len(hi):3d} acc={np.mean(hi):.3f}")
    accs = [tab[s]["acc"] for s in tab]
    for key, label in (("hand", "hand rate"), ("T", "frames"), ("sw", "shoulder width"), ("ldom", "left-dominant share")):
        rho, p = spearmanr([tab[s][key] for s in tab], accs)
        print(f"   Spearman(signer acc, {label}): rho={rho:.2f} (p={p:.2f}, n={len(tab)} signers)")
    print()

    print("C. Most frequent confusions (true -> pred), and how close the two classes are in landmark space")
    cls_seqs = defaultdict(list)
    for x in R:
        cls_seqs[x["true"]].append(prepare_input(np.load(x["path"]).astype(np.float32), 60).reshape(-1))
    names = sorted(cls_seqs)
    mean_t = {c: np.mean(cls_seqs[c], axis=0) for c in cls_seqs}

    def dist(a, b, sl):
        return float(np.linalg.norm(mean_t[a].reshape(60, -1)[:, sl] - mean_t[b].reshape(60, -1)[:, sl]))
    slices = {"all": slice(0, 258), "hands": slice(POSE_DIM, 258), "pose": slice(0, POSE_DIM)}
    all_d = {k: [dist(a, b, sl) for i, a in enumerate(names) for b in names[i + 1:]] for k, sl in slices.items()}
    conf = Counter()
    for x in R:
        for pr, ok in zip(x["pred"], x["ok"]):
            if not ok:
                conf[(x["true"], pr)] += 1
    print("   true -> pred        count  reverse  landmark-distance percentile among all class pairs (low = very similar)")
    for (a, b), n in conf.most_common(15):
        if b.startswith("_"):
            continue
        pct = {k: 100 * np.mean(np.array(all_d[k]) <= dist(a, b, sl)) for k, sl in slices.items()}
        print(f"   {a:11s}-> {b:11s} {n:4d}   {conf.get((b, a), 0):4d}    all {pct['all']:4.0f}%  hands {pct['hands']:4.0f}%  pose {pct['pose']:4.0f}%")
    print()

    print("D. Can wrong predictions be rejected by confidence? (each seed's prediction counted separately)")
    confs = np.array([c for x in R for c in x["conf"]])
    oks = np.array([o for x in R for o in x["ok"]])
    print(f"   mean confidence: correct {confs[oks].mean():.3f} | wrong {confs[~oks].mean():.3f}")
    print("   threshold  coverage  accuracy-on-accepted  errors-remaining")
    for th in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = confs >= th
        print(f"   {th:>8.1f}  {m.mean():8.3f}  {oks[m].mean():20.3f}  {(~oks[m]).sum() / len(oks):16.3f}")
    print()

    print("E. Sink classes (predicted more often than they occur; precision shows the damage)")
    pred_n, true_n, hit = Counter(), Counter(), Counter()
    for x in R:
        for pr, ok in zip(x["pred"], x["ok"]):
            pred_n[pr] += 1
            true_n[x["true"]] += 1
            hit[pr] += ok
    sinks = sorted(pred_n, key=lambda c: pred_n[c] / max(true_n[c], 1), reverse=True)[:8]
    for c in sinks:
        print(f"   {c:12s} predicted {pred_n[c]:4d} vs occurs {true_n[c]:4d}  precision {hit[c] / pred_n[c]:.3f}")
    errs = Counter()
    for x in R:
        errs[x["true"]] += len(x["ok"]) - sum(x["ok"])
    tot = sum(errs.values())
    top10 = sum(n for _, n in errs.most_common(10))
    print(f"\n   errors are concentrated: the 10 worst classes hold {top10 / tot:.0%} of all errors "
          f"({len(errs)} classes total)")


if __name__ == "__main__":
    main()
