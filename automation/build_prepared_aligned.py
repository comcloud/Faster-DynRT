#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def has_npy(tensor_root: Path, sid: str) -> bool:
    return (tensor_root / f"{sid}.npy").exists()


def align_split(prepared_root: Path, tensor_root: Path, split: str) -> Tuple[Dict[str, List], Dict[str, int], List[str]]:
    ids = [str(x) for x in load_pickle(prepared_root / f"{split}_id")]
    txt = load_pickle(prepared_root / f"{split}_text")
    lbl = load_pickle(prepared_root / f"{split}_labels")
    att_lines = (prepared_root / "att" / f"{split}_att.txt").read_text(encoding="utf-8", errors="ignore").splitlines()

    n = min(len(ids), len(txt), len(lbl), len(att_lines))
    keep_idx = [i for i in range(n) if has_npy(tensor_root, ids[i])]
    miss_ids = [ids[i] for i in range(n) if i not in set(keep_idx)]

    out = {
        "id": [ids[i] for i in keep_idx],
        "text": [txt[i] for i in keep_idx],
        "labels": [lbl[i] for i in keep_idx],
        "att": [att_lines[i] for i in keep_idx],
    }
    stat = {"raw": n, "kept": len(keep_idx), "dropped": n - len(keep_idx)}
    return out, stat, miss_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Build aligned prepared dataset that has matching npy tensors.")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--prepared-root", default="input/prepared")
    ap.add_argument("--tensor-root", required=True)
    ap.add_argument("--output-root", default="input/prepared_aligned")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    prepared_root = (root / args.prepared_root).resolve()
    tensor_root = Path(args.tensor_root).resolve()
    output_root = (root / args.output_root).resolve()
    (output_root / "att").mkdir(parents=True, exist_ok=True)

    report = {"splits": {}, "total_raw": 0, "total_kept": 0, "total_dropped": 0}
    all_missing = {}
    for split in ["train", "valid", "test"]:
        out, stat, miss_ids = align_split(prepared_root, tensor_root, split)
        dump_pickle(output_root / f"{split}_id", out["id"])
        dump_pickle(output_root / f"{split}_text", out["text"])
        dump_pickle(output_root / f"{split}_labels", out["labels"])
        (output_root / "att" / f"{split}_att.txt").write_text("\n".join(out["att"]) + ("\n" if out["att"] else ""), encoding="utf-8")

        report["splits"][split] = stat
        report["total_raw"] += stat["raw"]
        report["total_kept"] += stat["kept"]
        report["total_dropped"] += stat["dropped"]
        all_missing[split] = miss_ids

    report_path = output_root / "alignment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_path = output_root / "missing_ids_by_split.json"
    missing_path.write_text(json.dumps(all_missing, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))
    print(f"aligned dataset: {output_root}")
    print(f"report: {report_path}")
    print(f"missing ids: {missing_path}")


if __name__ == "__main__":
    main()
