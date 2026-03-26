#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock train entry for pipeline smoke tests.")
    parser.add_argument("config")
    parser.add_argument("--valid-f1", type=float, default=None)
    parser.add_argument("--test-f1", type=float, default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    stamp = dt.datetime.now().strftime("%m-%d-%H_%M_%S_%f")
    exp_dir = Path("exp") / stamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    valid_f1 = args.valid_f1
    test_f1 = args.test_f1
    if valid_f1 is None:
        valid_f1 = float(cfg.get("mock", {}).get("valid_f1", 0.80))
    if test_f1 is None:
        test_f1 = float(cfg.get("mock", {}).get("test_f1", max(valid_f1 - 0.01, 0.0)))

    lines = [
        f"valid: F1: {valid_f1:.4f}, Precision: {valid_f1:.4f}, Recall : {valid_f1:.4f}, Accuracy: {valid_f1:.4f}, Loss: 0.1000.",
        f"test: F1: {test_f1:.4f}, Precision: {test_f1:.4f}, Recall : {test_f1:.4f}, Accuracy: {test_f1:.4f}, Loss: 0.1000.",
    ]
    (exp_dir / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[mock_train] wrote {exp_dir / 'log.txt'}")


if __name__ == "__main__":
    main()
