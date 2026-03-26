#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_CONTEXT_FILES = [
    "model/DynRT.py",
    "model/attention/BridgeInfoLayer.py",
    "model/attention/CrossModalTransformerLayer.py",
    "model/attention/MultimodalFusionLayer.py",
    "model/attention/GuideAttentionLayer.py",
    "model/attention/TraditionalAttentionLayer.py",
]


def parse_target_files(instruction_text: str) -> list[str]:
    targets: list[str] = []
    in_targets = False
    for raw_line in instruction_text.splitlines():
        line = raw_line.strip()
        if line == "Target files:":
            in_targets = True
            continue
        if in_targets and line.startswith("- "):
            targets.append(line[2:].strip())
            continue
        if in_targets and line and not line.startswith("- "):
            in_targets = False
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent adapter for experiment pipeline.")
    parser.add_argument("--workspace", required=True, help="Project workspace path")
    parser.add_argument("--instruction-file", required=True, help="Instruction file path")
    parser.add_argument("--task-name", required=True, help="Task name")
    args = parser.parse_args()

    workspace = str(Path(args.workspace).resolve())
    instruction_file = str(Path(args.instruction_file).resolve())
    task_name = args.task_name
    instruction_text = Path(instruction_file).read_text(encoding="utf-8", errors="ignore")

    backend = os.environ.get("AGENT_BACKEND", "noop").strip().lower()
    if backend == "noop":
        print(f"[agent_adapter] backend=noop, skip real agent call for task: {task_name}")
        return 0

    if backend == "openai_api":
        env = os.environ.copy()
        if not env.get("AGENT_CONTEXT_FILES", "").strip():
            targets = parse_target_files(instruction_text)
            merged = []
            for p in DEFAULT_CONTEXT_FILES + targets:
                if p and p not in merged:
                    merged.append(p)
            env["AGENT_CONTEXT_FILES"] = ",".join(merged)
        cmd = [
            sys.executable,
            "automation/agent_openai_patch.py",
            "--workspace",
            workspace,
            "--instruction-file",
            instruction_file,
            "--task-name",
            task_name,
        ]
        print("[agent_adapter] run:", " ".join(shlex.quote(x) for x in cmd))
        result = subprocess.run(cmd, cwd=workspace, env=env, check=False)
        return result.returncode

    if backend != "shell":
        print(f"[agent_adapter] unsupported AGENT_BACKEND={backend}, expected noop/shell/openai_api", file=sys.stderr)
        return 2

    template = os.environ.get("AGENT_SHELL_TEMPLATE", "").strip()
    if not template:
        print("[agent_adapter] AGENT_SHELL_TEMPLATE is required when AGENT_BACKEND=shell", file=sys.stderr)
        return 2

    rendered = template.format(
        workspace=workspace,
        instruction_file=instruction_file,
        instruction=instruction_text,
        task_name=task_name,
    )
    cmd = shlex.split(rendered)
    print("[agent_adapter] run:", " ".join(shlex.quote(x) for x in cmd))
    result = subprocess.run(cmd, cwd=workspace, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
