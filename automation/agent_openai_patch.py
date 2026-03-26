#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def parse_instruction(instruction_path: Path) -> Dict[str, object]:
    raw = instruction_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()
    task = ""
    instruction = []
    targets: List[str] = []
    mode = "head"
    for line in lines:
        if line.startswith("Task:"):
            task = line.split("Task:", 1)[1].strip()
            continue
        if line.strip() == "Target files:":
            mode = "targets"
            continue
        if mode == "targets":
            if line.strip().startswith("- "):
                targets.append(line.strip()[2:].strip())
            continue
        if line.strip():
            instruction.append(line)
    return {
        "task": task,
        "instruction": "\n".join(instruction).strip(),
        "targets": targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Use OpenAI model to modify target files directly.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--task-name", required=True)
    args = parser.parse_args()

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("KEY", "").strip()
        or os.environ.get("API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (or KEY/API_KEY) is required for AGENT_BACKEND=openai_api")
    model = (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("MODEL", "").strip()
        or "gpt-5.4-mini"
    )
    if ":" in model:
        # Accept provider-prefixed style like "gemini:gemini-2.5-flash".
        model = model.split(":", 1)[1].strip()
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "").strip()
        or os.environ.get("BASE_URL", "").strip()
    )
    if base_url.endswith("`"):
        base_url = base_url[:-1]
    if base_url:
        base_url = base_url.rstrip("/")
        if "generativelanguage.googleapis.com" in base_url and "/openai" not in base_url:
            # Gemini OpenAI compatibility endpoint.
            base_url = base_url + "/v1beta/openai"

    workspace = Path(args.workspace).resolve()
    instruction_path = Path(args.instruction_file).resolve()
    parsed = parse_instruction(instruction_path)
    targets: List[str] = parsed["targets"]  # type: ignore[assignment]
    if not targets:
        raise RuntimeError("No target files in instruction file.")

    file_payload = []
    for rel_path in targets:
        abs_path = (workspace / rel_path).resolve()
        if not abs_path.exists():
            raise FileNotFoundError(f"Target file not found: {abs_path}")
        if workspace not in abs_path.parents and abs_path != workspace:
            raise RuntimeError(f"Target out of workspace: {abs_path}")
        content = abs_path.read_text(encoding="utf-8", errors="ignore")
        file_payload.append({"path": rel_path, "content": content})

    system_prompt = (
        "You are a careful coding agent. "
        "Modify only the provided target files and preserve public interfaces unless asked otherwise. "
        "Return strict JSON: {\"updates\":[{\"path\":\"...\",\"content\":\"...\"}],\"summary\":\"...\"}. "
        "No markdown."
    )
    user_prompt = {
        "task_name": args.task_name,
        "instruction": parsed["instruction"],
        "target_files": file_payload,
        "constraints": [
            "Do not change file paths.",
            "Return full updated content for each file that changes.",
            "Only include files from target_files in updates.",
        ],
    }

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    def _create_completion(model_name: str):
        return client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        )

    used_model = model
    try:
        resp = _create_completion(used_model)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "not found" in msg and "generativelanguage.googleapis.com" in (base_url or ""):
            fallback = "gemini-2.5-flash"
            resp = _create_completion(fallback)
            used_model = fallback
        else:
            raise
    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError("Model returned empty response.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Compatible with providers that wrap JSON in markdown fences.
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        payload = json.loads(m.group(0))
    updates = payload.get("updates", [])
    if not isinstance(updates, list):
        raise RuntimeError("Invalid response: updates must be list.")

    allowed = set(targets)
    changed: List[str] = []
    for upd in updates:
        rel_path = upd.get("path")
        content = upd.get("content")
        if not isinstance(rel_path, str) or not isinstance(content, str):
            raise RuntimeError("Invalid update item shape.")
        if rel_path not in allowed:
            raise RuntimeError(f"Update includes non-target file: {rel_path}")
        abs_path = (workspace / rel_path).resolve()
        abs_path.write_text(content, encoding="utf-8")
        changed.append(rel_path)

    print(
        json.dumps(
            {
                "task": args.task_name,
                "used_model": used_model,
                "changed_files": changed,
                "summary": payload.get("summary", ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
