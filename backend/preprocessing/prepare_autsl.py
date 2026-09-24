"""Select AUTSL videos from an extracted shard and stage them for training.

Reads train_labels.csv (sample -> ClassId) and SignList_ClassId_TR_EN.csv
(ClassId -> Turkish name), scans a folder of extracted *_color.mp4 files,
picks the classes with the most samples and copies them into
data/raw_videos/<turkish_name>/ ready for extract_landmarks.py.

Usage:
    python backend/preprocessing/prepare_autsl.py --shard-dir backend/data/autsl/shard_001 --top 5
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # backend/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True, help="folder with extracted AUTSL videos")
    parser.add_argument("--autsl-dir", type=Path, default=root / "data" / "autsl")
    parser.add_argument("--output", type=Path, default=root / "data" / "raw_videos")
    parser.add_argument("--top", type=int, default=5, help="number of classes to stage")
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--classes", type=str, default=None,
                        help="comma-separated Turkish class names (overrides --top)")
    parser.add_argument("--signer", type=str, default=None,
                        help="only stage videos of this signer, e.g. signer0")
    parser.add_argument("--exclude-signer", type=str, default=None,
                        help="skip videos of this signer (e.g. the held-out test signer)")
    parser.add_argument("--labels-csv", type=str, default="train_labels.csv",
                        help="label file under --autsl-dir (train_labels.csv / validation_labels.csv / test_labels.csv)")
    args = parser.parse_args()

    with open(args.autsl_dir / "SignList_ClassId_TR_EN.csv", encoding="utf-8-sig") as f:
        class_names = {int(r["ClassId"]): r["TR"] for r in csv.DictReader(f)}
    with open(args.autsl_dir / args.labels_csv, encoding="utf-8-sig") as f:
        sample_class = {row[0]: int(row[1]) for row in csv.reader(f) if len(row) == 2}

    videos = sorted(args.shard_dir.rglob("*_color.mp4"))
    if args.signer:
        videos = [v for v in videos if v.stem.startswith(args.signer + "_")]
    if args.exclude_signer:
        videos = [v for v in videos if not v.stem.startswith(args.exclude_signer + "_")]
    if not videos:
        raise SystemExit(f"No matching *_color.mp4 found under {args.shard_dir}")
    by_class: dict[int, list[Path]] = defaultdict(list)
    unknown = 0
    for v in videos:
        sample_id = v.stem.removesuffix("_color")
        cid = sample_class.get(sample_id)
        if cid is None:
            unknown += 1
            continue
        by_class[cid].append(v)
    print(f"{len(videos)} videos in shard, {len(by_class)} classes, {unknown} without label")

    if args.classes:
        wanted = {n.strip() for n in args.classes.split(",") if n.strip()}
        name_to_cid = {v: k for k, v in class_names.items()}
        missing = wanted - set(name_to_cid)
        if missing:
            raise SystemExit(f"Unknown class names: {sorted(missing)}")
        ranked = [(name_to_cid[n], by_class.get(name_to_cid[n], [])) for n in sorted(wanted)]
        ranked = [(cid, vids) for cid, vids in ranked if vids]
    else:
        ranked = sorted(by_class.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.top]
    for cid, vids in ranked:
        name = class_names.get(cid, f"class_{cid}")
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        for v in vids[: args.max_per_class]:
            dst = out / v.name
            if not dst.exists():
                shutil.copy2(v, dst)
        signers = {v.stem.split("_")[0] for v in vids}
        print(f"  [{cid:3d}] {name}: {min(len(vids), args.max_per_class)} videos, {len(signers)} signer(s) -> {out}")
    print("Done. Next: run extract_landmarks.py")


if __name__ == "__main__":
    main()
