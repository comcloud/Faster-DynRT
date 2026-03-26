#!/usr/bin/env python3
import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

# make project root importable when script runs from automation/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from input.loader_img import loader_img
from input.loader_label import loader_label
from input.loader_text import loader_text


def dump_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


class SimpleTokenizer:
    def __call__(self, text: str):
        # deterministic lightweight tokenizer: mimic HF return shape
        ids = [101]
        for ch in text[:98]:
            ids.append((ord(ch) % 200) + 1)
        ids.append(102)
        return {"input_ids": ids}


@dataclass
class SamplePack:
    train_idx: List[int]
    valid_idx: List[int]
    test_idx: List[int]


def pick_indices(prepared_root: Path, tensor_root: Path, per_split: int, seed: int) -> SamplePack:
    random.seed(seed)
    out = {}
    for split in ["train", "valid", "test"]:
        ids = [str(x) for x in load_pickle(prepared_root / f"{split}_id")]
        cand = [i for i, sid in enumerate(ids) if (tensor_root / f"{sid}.npy").exists()]
        if len(cand) < per_split:
            raise RuntimeError(f"Not enough samples with npy for split={split}, got={len(cand)}")
        random.shuffle(cand)
        out[split] = sorted(cand[:per_split])
    return SamplePack(train_idx=out["train"], valid_idx=out["valid"], test_idx=out["test"])


def build_subset(prepared_root: Path, subset_root: Path, pack: SamplePack) -> None:
    subset_root.mkdir(parents=True, exist_ok=True)
    (subset_root / "att").mkdir(parents=True, exist_ok=True)
    for split, idxs in [("train", pack.train_idx), ("valid", pack.valid_idx), ("test", pack.test_idx)]:
        ids = load_pickle(prepared_root / f"{split}_id")
        txt = load_pickle(prepared_root / f"{split}_text")
        y = load_pickle(prepared_root / f"{split}_labels")
        dump_pickle(subset_root / f"{split}_id", [ids[i] for i in idxs])
        dump_pickle(subset_root / f"{split}_text", [txt[i] for i in idxs])
        dump_pickle(subset_root / f"{split}_labels", [y[i] for i in idxs])

        att_lines = (prepared_root / "att" / f"{split}_att.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
        selected = [att_lines[i] for i in idxs]
        (subset_root / "att" / f"{split}_att.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")


class SmokeDataset(torch.utils.data.Dataset):
    def __init__(self, loaders: Dict[str, object], mode: str):
        self.loaders = loaders
        self.mode = mode
        self.length = None
        for loader in self.loaders.values():
            l = loader.getlength(mode)
            self.length = l if self.length is None else min(self.length, l)
        self.length = int(self.length or 0)

    def __getitem__(self, idx):
        out = {}
        for key, loader in self.loaders.items():
            loader.get(out, self.mode, idx)
        return out

    def __len__(self):
        return self.length


class TinyFusionModel(nn.Module):
    def __init__(self, seq_len: int = 100):
        super().__init__()
        self.text_proj = nn.Linear(seq_len, 64)
        self.att_proj = nn.Linear(seq_len, 32)
        self.img_proj = nn.Linear(3 * 224 * 224, 64)
        self.cls = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64 + 32 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, batch):
        text = batch["text"].float()
        att = batch["att"].float()
        img = batch["img"].float().view(batch["img"].size(0), -1)
        z_text = self.text_proj(text)
        z_att = self.att_proj(att)
        z_img = self.img_proj(img)
        z = torch.cat([z_text, z_att, z_img], dim=1)
        scores = self.cls(z)
        return scores


def compute_acc_micro_macro_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return {"acc": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}
    labels = sorted(set(y_true) | set(y_pred))
    tp_total = fp_total = fn_total = 0
    f1_list = []
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
    for c in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        f1_list.append(f1)
    p_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    r_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_micro = (2 * p_micro * r_micro / (p_micro + r_micro)) if (p_micro + r_micro) > 0 else 0.0
    return {
        "acc": float(correct / n),
        "micro_f1": float(f1_micro),
        "macro_f1": float(sum(f1_list) / len(f1_list)),
    }


def eval_split(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            scores = model(batch)
            preds = scores.argmax(dim=1)
            ys.extend(batch["label"].cpu().tolist())
            ps.extend(preds.cpu().tolist())
    m = compute_acc_micro_macro_f1(ys, ps)
    m["count"] = len(ys)
    return m


def main():
    ap = argparse.ArgumentParser(description="CPU e2e smoke test for full data->model->train/valid/test flow.")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--prepared-root", default="input/prepared")
    ap.add_argument("--image-tensor-root", required=True)
    ap.add_argument("--per-split", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    prepared_root = (root / args.prepared_root).resolve()
    tensor_root = Path(args.image_tensor_root).resolve()
    subset_root = root / "auto_runs" / "e2e_smoke_subset"
    report_path = root / "auto_runs" / "e2e_smoke_report.json"

    if not tensor_root.exists():
        raise RuntimeError(f"image tensor root not found: {tensor_root}")

    pack = pick_indices(prepared_root, tensor_root, args.per_split, args.seed)
    build_subset(prepared_root, subset_root, pack)

    tok = SimpleTokenizer()
    inputs = {"tokenizer_roberta": tok}
    l_text = loader_text()
    l_img = loader_img()
    l_label = loader_label()
    l_text.prepare(inputs, {
        "source": "prepared",
        "data_path": str(subset_root) + "/",
        "att_file_path": str(subset_root / "att") + "/",
        "len": 100,
        "pad": 1,
        "mould": "A photo containing the {att_0}, {att_1}, {att_2}, {att_3} and {att_4}",
    })
    l_img.prepare({}, {
        "source": "prepared",
        "data_path": str(subset_root) + "/",
        "transform_image": str(tensor_root) + "/",
        "image_root": "",
        "image_resize": 224,
    })
    l_label.prepare({}, {
        "source": "prepared",
        "data_path": str(subset_root) + "/",
        "test_label": True,
    })

    loaders = {"text": l_text, "img": l_img, "label": l_label}
    dls = {
        split: torch.utils.data.DataLoader(
            SmokeDataset(loaders, split),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            drop_last=False,
        )
        for split in ["train", "valid", "test"]
    }

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = TinyFusionModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_steps = 0
    for _ in range(args.epochs):
        model.train()
        for batch in dls["train"]:
            for k in batch:
                batch[k] = batch[k].to(device)
            scores = model(batch)
            loss = loss_fn(scores, batch["label"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_steps += 1

    result = {
        "ok": True,
        "device": str(device),
        "train_steps": train_steps,
        "subset_sizes": {"train": len(pack.train_idx), "valid": len(pack.valid_idx), "test": len(pack.test_idx)},
        "metrics": {
            "valid": eval_split(model, dls["valid"], device),
            "test": eval_split(model, dls["test"], device),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"e2e smoke ok: {report_path}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
