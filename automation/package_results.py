#!/usr/bin/env python3
import argparse
import json
import tarfile
from pathlib import Path
from typing import List


def collect_paths(run_root: Path) -> List[Path]:
    paths = [run_root]
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary:
            exp_dir = row.get("exp_dir")
            if exp_dir:
                p = Path(exp_dir).resolve()
                if p.exists():
                    paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack auto experiment outputs and linked exp logs/checkpoints.")
    parser.add_argument("--run-root", required=True, help="Path like auto_runs/module10_msd_local")
    parser.add_argument("--output", required=True, help="Output tar.gz path")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    include_paths = collect_paths(run_root)
    with tarfile.open(output, "w:gz") as tar:
        for p in include_paths:
            tar.add(str(p), arcname=str(p.name))

    print(f"packed: {output}")
    print("includes:")
    for p in include_paths:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
