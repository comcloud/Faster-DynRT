#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def read_ids_txt(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def read_ids_pickle(path: Path) -> List[str]:
    with path.open("rb") as f:
        return [str(x) for x in pickle.load(f)]


def resolve_image(image_root: Path, sample_id: str) -> Path:
    stem = Path(sample_id).stem
    candidates = [
        image_root / sample_id,
        image_root / f"{stem}.jpg",
        image_root / f"{stem}.png",
        image_root / f"{stem}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for id={sample_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute image tensors (.npy) from dataset id files.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--tensor-root", required=True)
    parser.add_argument("--source", default="prepared", choices=["prepared", "bully", "msd2"])
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    prepared_dir = root / "input" / "prepared"
    bully_dir = root / "input" / "bully"
    msd2_dir = root / "input" / "MSD2"
    image_root = Path(args.image_root).resolve()
    tensor_root = Path(args.tensor_root).resolve()
    tensor_root.mkdir(parents=True, exist_ok=True)

    all_ids: List[str] = []
    if args.source == "prepared":
        for split in ["train", "valid", "test"]:
            all_ids.extend(read_ids_pickle(prepared_dir / f"{split}_id"))
    elif args.source == "bully":
        for split in ["train", "valid", "test"]:
            all_ids.extend(read_ids_txt(bully_dir / f"{split}_id.txt"))
    else:
        import json
        for split in ["train", "valid", "test"]:
            rows = json.loads((msd2_dir / f"{split}.json").read_text(encoding="utf-8"))
            all_ids.extend([str(item["image_id"]) for item in rows])
    unique_ids = sorted(set(all_ids))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    ok = 0
    missing: List[Tuple[str, str]] = []
    for sid in unique_ids:
        out_path = tensor_root / f"{Path(sid).stem}.npy"
        if out_path.exists():
            ok += 1
            continue
        try:
            img_path = resolve_image(image_root, sid)
            img = Image.open(img_path).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            arr = (arr - mean) / std
            np.save(out_path, arr.astype(np.float16))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            missing.append((sid, str(exc)))

    print(f"total_ids={len(unique_ids)} prepared={ok} missing={len(missing)}")
    if missing:
        miss_file = tensor_root / "missing_images.txt"
        miss_file.write_text("\n".join([f"{sid}\t{msg}" for sid, msg in missing]) + "\n", encoding="utf-8")
        print(f"missing details: {miss_file}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
