#!/usr/bin/env python3
import argparse
import copy
import csv
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_LINE_RE = re.compile(r"valid\s*:.*F1:\s*([0-9.]+)", re.IGNORECASE)
TEST_LINE_RE = re.compile(r"test\s*:.*F1:\s*([0-9.]+)", re.IGNORECASE)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_by_dot_path(payload: Dict[str, Any], dot_path: str, value: Any) -> None:
    parts = dot_path.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def list_exp_dirs(exp_root: Path) -> List[Path]:
    if not exp_root.exists():
        return []
    return sorted([p for p in exp_root.iterdir() if p.is_dir()])


def find_new_exp_dir(before: List[Path], after: List[Path]) -> Optional[Path]:
    before_set = {p.name for p in before}
    new_dirs = [p for p in after if p.name not in before_set]
    return new_dirs[-1] if new_dirs else None


def parse_log_metrics(log_file: Path) -> Tuple[Optional[float], Optional[float]]:
    if not log_file.exists():
        return None, None
    valid_f1 = None
    test_f1 = None
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        m_valid = VALID_LINE_RE.search(line)
        if m_valid:
            valid_f1 = float(m_valid.group(1))
        m_test = TEST_LINE_RE.search(line)
        if m_test:
            test_f1 = float(m_test.group(1))
    return valid_f1, test_f1


def parse_structured_metrics(exp_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    metrics_file = exp_dir / "metrics_history.json"
    best_file = exp_dir / "best_model_info.json"
    if metrics_file.exists():
        rows = json.loads(metrics_file.read_text(encoding="utf-8"))
        if rows:
            last = rows[-1]
            valid = last.get("valid", {})
            test = last.get("test", {})
            out.update(
                {
                    "valid_acc": valid.get("accuracy"),
                    "valid_f1_micro": valid.get("f1_micro"),
                    "valid_f1_macro": valid.get("f1_macro"),
                    "test_acc": test.get("accuracy"),
                    "test_f1_micro": test.get("f1_micro"),
                    "test_f1_macro": test.get("f1_macro"),
                }
            )
    if best_file.exists():
        best = json.loads(best_file.read_text(encoding="utf-8"))
        out.update(
            {
                "best_metric_key": best.get("best_metric_key"),
                "best_metric_value": best.get("best_metric_value"),
                "best_epoch": best.get("best_epoch"),
                "best_checkpoint_path": best.get("checkpoint_path"),
            }
        )
    return out


def prune_checkpoint_files(exp_dir: Path) -> None:
    # Support both legacy "checkpoint/" and current "checkpoints/" layouts.
    # Keep only one best checkpoint per layout to limit disk usage.
    layouts = [
        ("checkpoints", "model_best.pth.tar"),
        ("checkpoint", "new_model_best.pth.tar"),
    ]
    for dirname, keep_name in layouts:
        ckpt_dir = exp_dir / dirname
        if not ckpt_dir.exists():
            continue
        for p in ckpt_dir.glob("*.pth*"):
            if p.name == keep_name:
                continue
            p.unlink(missing_ok=True)


def prune_exp_history(
    exp_root: Path,
    keep_last: int,
    pinned_dirs: List[str],
    prune_checkpoints: bool,
) -> None:
    if keep_last < 1 or not exp_root.exists():
        return
    all_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    pinned = {x for x in pinned_dirs if x}
    survivors = set(p.name for p in all_dirs[-keep_last:]) | pinned
    for p in all_dirs:
        if p.name in survivors:
            if prune_checkpoints:
                prune_checkpoint_files(p)
            continue
        shutil.rmtree(p, ignore_errors=True)


def run_command(
    cmd: List[str],
    cwd: Path,
    env: Optional[Dict[str, str]],
    log_path: Path,
    dry_run: bool,
    timeout_sec: Optional[int],
) -> int:
    if dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("DRY-RUN: " + " ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")
        print("[DRY-RUN]", " ".join(shlex.quote(x) for x in cmd))
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return process.returncode


def render_agent_command(template: str, mapping: Dict[str, str]) -> List[str]:
    rendered = template.format(**mapping)
    return shlex.split(rendered)


def execute_agent_task(
    project_root: Path,
    experiment_dir: Path,
    agent_cfg: Dict[str, Any],
    task_cfg: Dict[str, Any],
    dry_run: bool,
) -> int:
    command_template = agent_cfg.get("command", "").strip()
    if not command_template:
        return 0

    task_name = task_cfg["name"]
    instruction = task_cfg.get("instruction", "").strip()
    target_files = task_cfg.get("target_files", [])
    instruction_file = experiment_dir / f"{task_name}.instruction.txt"
    instruction_lines = [f"Task: {task_name}", "", instruction]
    if target_files:
        instruction_lines.extend(["", "Target files:"])
        instruction_lines.extend([f"- {file_path}" for file_path in target_files])
    instruction_file.write_text("\n".join(instruction_lines).strip() + "\n", encoding="utf-8")

    mapping = {
        "workspace": str(project_root),
        "instruction_file": str(instruction_file),
        "instruction": instruction,
        "task_name": task_name,
    }
    cmd = render_agent_command(command_template, mapping)
    timeout_sec = agent_cfg.get("timeout_sec")
    env = os.environ.copy()
    env.update({k: str(v) for k, v in task_cfg.get("env", {}).items()})
    return run_command(
        cmd=cmd,
        cwd=project_root,
        env=env,
        log_path=experiment_dir / "agent.log",
        dry_run=dry_run,
        timeout_sec=timeout_sec,
    )


def build_experiment_config(
    base_cfg: Dict[str, Any],
    overrides: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("info", {}).setdefault("log", {})
    cfg["info"]["name"] = experiment_name
    for dot_path, value in overrides.items():
        set_by_dot_path(cfg, dot_path, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent-driven DynRT experiments automatically.")
    parser.add_argument("--plan", required=True, help="Path to experiment plan JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, no execution.")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)

    project_root = Path(plan.get("project_root", plan_path.parent.parent)).resolve()
    base_config_path = (project_root / plan["base_config"]).resolve() if not os.path.isabs(plan["base_config"]) else Path(plan["base_config"]).resolve()
    base_cfg = load_json(base_config_path)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (project_root / plan.get("output_dir", f"auto_runs/{stamp}")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    python_bin = plan.get("python_bin", sys.executable)
    train_script = plan.get("train_script", "train.py")
    exp_root = (project_root / "exp").resolve()

    agent_cfg = plan.get("agent", {})
    agent_tasks = {task["name"]: task for task in plan.get("agent_tasks", [])}
    cleanup_cfg = plan.get("cleanup", {})

    summary: List[Dict[str, Any]] = []
    global_best: Optional[Dict[str, Any]] = None
    experiments = plan.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in plan.")

    print(f"Plan: {plan_path}")
    print(f"Project root: {project_root}")
    print(f"Run root: {run_root}")
    print(f"Experiments: {len(experiments)}")

    for idx, exp_cfg in enumerate(experiments, start=1):
        exp_name = exp_cfg["name"]
        exp_dir = run_root / f"{idx:02d}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{idx}/{len(experiments)}] {exp_name}")

        agent_task_name = exp_cfg.get("agent_task")
        agent_rc = 0
        agent_task_cfg = None
        if agent_task_name:
            if agent_task_name not in agent_tasks:
                raise ValueError(f"agent_task '{agent_task_name}' not found for experiment '{exp_name}'")
            agent_task_cfg = agent_tasks[agent_task_name]
            print(f"  - running agent task: {agent_task_name}")
            agent_rc = execute_agent_task(
                project_root=project_root,
                experiment_dir=exp_dir,
                agent_cfg=agent_cfg,
                task_cfg=agent_task_cfg,
                dry_run=args.dry_run,
            )
            if agent_rc != 0:
                print(f"  - agent task failed with code {agent_rc}, skip training")
                summary.append(
                    {
                        "name": exp_name,
                        "agent_task": agent_task_name,
                        "agent_return_code": agent_rc,
                        "train_return_code": None,
                        "valid_f1": None,
                        "test_f1": None,
                        "exp_dir": None,
                    }
                )
                continue

        overrides = exp_cfg.get("config_overrides", {})
        config_payload = build_experiment_config(base_cfg, overrides, experiment_name=exp_name)
        config_file = exp_dir / "config.json"
        save_json(config_file, config_payload)

        before_dirs = list_exp_dirs(exp_root)
        cmd = [python_bin, train_script, str(config_file)]
        cmd.extend(exp_cfg.get("train_args", []))
        env = os.environ.copy()
        env.update({k: str(v) for k, v in exp_cfg.get("env", {}).items()})
        print("  - running training")
        train_rc = run_command(
            cmd=cmd,
            cwd=project_root,
            env=env,
            log_path=exp_dir / "train.log",
            dry_run=args.dry_run,
            timeout_sec=exp_cfg.get("timeout_sec"),
        )
        after_dirs = list_exp_dirs(exp_root)
        created_exp_dir = find_new_exp_dir(before_dirs, after_dirs)
        valid_f1 = None
        test_f1 = None
        if created_exp_dir is not None:
            valid_f1, test_f1 = parse_log_metrics(created_exp_dir / "log.txt")

        row = {
            "name": exp_name,
            "agent_task": agent_task_name,
            "agent_instruction": (agent_task_cfg or {}).get("instruction") if agent_task_cfg else None,
            "agent_target_files": (agent_task_cfg or {}).get("target_files") if agent_task_cfg else None,
            "agent_return_code": agent_rc,
            "train_return_code": train_rc,
            "valid_f1": valid_f1,
            "test_f1": test_f1,
            "exp_dir": str(created_exp_dir) if created_exp_dir else None,
            "config_file": str(config_file),
        }
        if created_exp_dir is not None:
            row.update(parse_structured_metrics(created_exp_dir))
        summary.append(row)

        best_key = row.get("best_metric_key") or "valid_f1_macro"
        best_val = row.get("best_metric_value")
        if best_val is None:
            best_val = row.get("valid_f1_macro")
        if best_val is None:
            best_val = row.get("valid_f1")
        try:
            best_val_f = float(best_val) if best_val is not None else None
        except Exception:  # noqa: BLE001
            best_val_f = None
        if best_val_f is not None:
            if global_best is None or best_val_f > float(global_best["metric_value"]):
                global_best = {
                    "metric_key": best_key,
                    "metric_value": best_val_f,
                    "experiment_name": exp_name,
                    "exp_dir": row.get("exp_dir"),
                    "best_checkpoint_path": row.get("best_checkpoint_path"),
                    "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
                with (run_root / "best_model.json").open("w", encoding="utf-8") as f:
                    json.dump(global_best, f, ensure_ascii=False, indent=2)

    summary_json = run_root / "summary.json"
    summary_csv = run_root / "summary.csv"
    summary_md = run_root / "summary.md"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "agent_task",
                "agent_instruction",
                "agent_target_files",
                "agent_return_code",
                "train_return_code",
                "valid_f1",
                "test_f1",
                "valid_acc",
                "valid_f1_micro",
                "valid_f1_macro",
                "test_acc",
                "test_f1_micro",
                "test_f1_macro",
                "best_metric_key",
                "best_metric_value",
                "best_epoch",
                "best_checkpoint_path",
                "exp_dir",
                "config_file",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    lines = [
        "# Auto Experiment Summary",
        "",
        "| name | agent_task | agent_rc | train_rc | valid_f1 | test_f1 | exp_dir |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            "| {name} | {agent_task} | {agent_return_code} | {train_return_code} | {valid_f1} | {test_f1} | {exp_dir} |".format(
                **{k: ("" if v is None else v) for k, v in row.items()}
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if cleanup_cfg.get("enabled", True):
        keep_last = int(cleanup_cfg.get("keep_exp_last", 20))
        prune_ckpt = bool(cleanup_cfg.get("prune_checkpoints", True))
        pinned_dirs: List[str] = []
        if global_best and global_best.get("exp_dir"):
            pinned_dirs.append(Path(str(global_best["exp_dir"])).name)
        for row in summary:
            exp_dir = row.get("exp_dir")
            if exp_dir:
                pinned_dirs.append(Path(str(exp_dir)).name)
        prune_exp_history(
            exp_root=exp_root,
            keep_last=keep_last,
            pinned_dirs=pinned_dirs,
            prune_checkpoints=prune_ckpt,
        )

    print("\nDone")
    print(f"- summary json: {summary_json}")
    print(f"- summary csv : {summary_csv}")
    print(f"- summary md  : {summary_md}")


if __name__ == "__main__":
    main()
