#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_ERROR_PATTERNS = [
    r"Traceback \(most recent call last\):",
    r"RuntimeError:",
    r"ValueError:",
    r"CUDA out of memory",
    r"AssertionError:",
    r"FileNotFoundError:",
    r"ConnectionError:",
]
VALID_ACC_RE = re.compile(r"valid\s*:.*accuracy:\s*([0-9.]+)", re.IGNORECASE)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_by_dot_path(payload: Dict[str, Any], dot_path: str, value: Any) -> None:
    parts = dot_path.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def build_experiment_config(base_cfg: Dict[str, Any], overrides: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("info", {}).setdefault("log", {})
    cfg["info"]["name"] = experiment_name
    for dot_path, value in overrides.items():
        set_by_dot_path(cfg, dot_path, value)
    return cfg


def list_exp_dirs(exp_root: Path) -> List[Path]:
    if not exp_root.exists():
        return []
    return sorted([p for p in exp_root.iterdir() if p.is_dir()])


def find_new_exp_dir(before: List[Path], after: List[Path]) -> Optional[Path]:
    before_set = {p.name for p in before}
    new_dirs = [p for p in after if p.name not in before_set]
    return new_dirs[-1] if new_dirs else None


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


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def pick_metric(row: Dict[str, Any], metric_key: str) -> Optional[float]:
    candidates = [metric_key, "valid_f1_macro", "best_metric_value", "valid_f1_micro", "valid_acc"]
    for key in candidates:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-max_lines:])


def parse_recent_valid_acc(log_path: Path, keep_last: int = 8) -> List[float]:
    if not log_path.exists():
        return []
    accs: List[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = VALID_ACC_RE.search(line)
        if m:
            try:
                accs.append(float(m.group(1)))
            except Exception:
                pass
    return accs[-keep_last:]


def parse_yaml_block_field(text: str, key: str) -> str:
    prefix = f"{key}:"
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.startswith(prefix):
            continue
        if line.strip() == f"{key}: |":
            out: List[str] = []
            for child in lines[idx + 1 :]:
                if child.startswith("  "):
                    out.append(child[2:])
                elif child.strip() == "":
                    out.append("")
                else:
                    break
            return "\n".join(out).strip()
        value = line.split(":", 1)[1].strip()
        return value
    return ""


def load_skill_content(skill_path: Path) -> str:
    raw = skill_path.read_text(encoding="utf-8", errors="ignore").strip()
    if skill_path.suffix.lower() in {".yaml", ".yml"}:
        extracted = parse_yaml_block_field(raw, "instructions")
        if extracted:
            return extracted
    return raw


def load_skill_guidance(project_root: Path, skill_files: List[str], max_chars: int) -> Tuple[str, List[str]]:
    chunks: List[str] = []
    loaded_paths: List[str] = []
    total_chars = 0
    for item in skill_files:
        rel = str(item).strip()
        if not rel:
            continue
        skill_path = Path(rel).expanduser()
        if not skill_path.is_absolute():
            skill_path = (project_root / rel).resolve()
        if not skill_path.exists() or not skill_path.is_file():
            continue
        content = load_skill_content(skill_path)
        if not content:
            continue
        block = f"[Skill: {skill_path.name}]\n{content}".strip()
        if max_chars > 0 and total_chars + len(block) > max_chars:
            remain = max_chars - total_chars
            if remain > 128:
                chunks.append(block[:remain])
                loaded_paths.append(str(skill_path))
            break
        chunks.append(block)
        loaded_paths.append(str(skill_path))
        total_chars += len(block)
    return ("\n\n".join(chunks).strip(), loaded_paths)


def trim_text_file(path: Path, keep_lines: int) -> None:
    if keep_lines <= 0 or not path.exists() or not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) <= keep_lines:
        return
    path.write_text("\n".join(lines[-keep_lines:]) + "\n", encoding="utf-8")


def prune_checkpoint_files(exp_dir: Path, keep_names: List[str]) -> None:
    keep = {x.strip() for x in keep_names if str(x).strip()}
    if not keep:
        keep = {"model_best.pth.tar", "new_model_best.pth.tar"}
    for dirname in ("checkpoints", "checkpoint"):
        ckpt_dir = exp_dir / dirname
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            continue
        for p in ckpt_dir.glob("*.pth*"):
            if p.name in keep:
                continue
            p.unlink(missing_ok=True)


def prune_exp_history(exp_root: Path, keep_last: int, pinned_dirs: List[str], prune_checkpoints: bool, keep_checkpoint_names: List[str]) -> None:
    if keep_last < 1 or not exp_root.exists():
        return
    all_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    pinned = {x for x in pinned_dirs if x}
    survivors = set(p.name for p in all_dirs[-keep_last:]) | pinned
    for p in all_dirs:
        if p.name in survivors:
            if prune_checkpoints:
                prune_checkpoint_files(p, keep_checkpoint_names)
            continue
        shutil.rmtree(p, ignore_errors=True)


def prune_run_attempt_history(run_root: Path, keep_last: int, pinned_dirs: List[str]) -> None:
    if keep_last < 1 or not run_root.exists():
        return
    attempt_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and "_attempt" in p.name])
    pinned = {x for x in pinned_dirs if x}
    survivors = set(p.name for p in attempt_dirs[-keep_last:]) | pinned
    for p in attempt_dirs:
        if p.name in survivors:
            continue
        shutil.rmtree(p, ignore_errors=True)


def cleanup_attempt_artifacts(
    attempt_dir: Path,
    keep_train_log_lines: int,
    keep_agent_log_lines: int,
    drop_instruction_files: bool,
    drop_agent_logs: bool,
) -> None:
    if not attempt_dir.exists():
        return
    trim_text_file(attempt_dir / "train.log", keep_train_log_lines)
    for p in attempt_dir.glob("*.agent.log"):
        if drop_agent_logs:
            p.unlink(missing_ok=True)
            continue
        trim_text_file(p, keep_agent_log_lines)
    if drop_instruction_files:
        for p in attempt_dir.glob("*.instruction.txt"):
            p.unlink(missing_ok=True)


def compile_patterns(patterns: List[str]) -> List[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def find_error_excerpt(log_path: Path, patterns: List[re.Pattern[str]]) -> Optional[str]:
    text = tail_text(log_path, max_lines=120)
    if not text:
        return None
    for pattern in patterns:
        if pattern.search(text):
            return text
    return None


def rel_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def collect_managed_files(plan: Dict[str, Any], fallback_targets: List[str]) -> List[str]:
    files = set(fallback_targets)
    for task in plan.get("agent_tasks", []):
        for rel_path in task.get("target_files", []):
            files.add(rel_path)
    return sorted(files)


def save_snapshot(workspace: Path, target_files: List[str], snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, Any]] = {}
    for rel_path in target_files:
        src = (workspace / rel_path).resolve()
        dst = (snapshot_dir / rel_path).resolve()
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest[rel_path] = {"exists": True}
        else:
            manifest[rel_path] = {"exists": False}
    save_json(snapshot_dir / "manifest.json", manifest)


def restore_snapshot(workspace: Path, snapshot_dir: Path) -> None:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = load_json(manifest_path)
    for rel_path, meta in manifest.items():
        dst = (workspace / rel_path).resolve()
        src = (snapshot_dir / rel_path).resolve()
        if meta.get("exists"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            dst.unlink(missing_ok=True)


def write_event(events_path: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pidfile(pidfile: Path) -> Optional[int]:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def write_pidfile(pidfile: Path, pid: int) -> None:
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(pid), encoding="utf-8")


def kill_pid_safely(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 8
    while time.time() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def kill_stale_train_by_config(config_file: Path) -> None:
    # Best-effort cleanup for orphaned train processes from previous controller runs.
    subprocess.run(["pkill", "-f", str(config_file)], check=False)


def render_agent_command(template: str, mapping: Dict[str, str]) -> List[str]:
    return shlex.split(template.format(**mapping))


def run_agent_task(
    workspace: Path,
    experiment_dir: Path,
    agent_cfg: Dict[str, Any],
    task_name: str,
    instruction: str,
    target_files: List[str],
    global_guidance: str = "",
) -> int:
    command_template = agent_cfg.get("command", "").strip()
    if not command_template:
        return 0
    instruction_file = experiment_dir / f"{task_name}.instruction.txt"
    lines = [f"Task: {task_name}"]
    if global_guidance.strip():
        lines.extend(["", "Global Guidance:", global_guidance.strip()])
    lines.extend(["", "Task Instruction:", instruction])
    if target_files:
        lines.extend(["", "Target files:"])
        lines.extend([f"- {p}" for p in target_files])
    instruction_file.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    mapping = {
        "workspace": str(workspace),
        "instruction_file": str(instruction_file),
        "instruction": instruction,
        "task_name": task_name,
    }
    cmd = render_agent_command(command_template, mapping)
    with (experiment_dir / f"{task_name}.agent.log").open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                env=os.environ.copy(),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=agent_cfg.get("timeout_sec"),
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            f.write("\n[background_controller] agent task timeout\n")
            f.flush()
            return 124


def launch_training(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    f.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
    f.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def stop_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def monitor_training(
    proc: subprocess.Popen[str],
    log_path: Path,
    error_patterns: List[re.Pattern[str]],
    poll_interval_sec: int,
    stall_timeout_sec: int,
    hard_timeout_sec: Optional[int],
    live_plateau_enabled: bool,
    live_plateau_window: int,
    live_plateau_epsilon: float,
) -> Dict[str, Any]:
    start_ts = time.time()
    last_mtime = log_path.stat().st_mtime if log_path.exists() else start_ts
    while True:
        rc = proc.poll()
        error_excerpt = find_error_excerpt(log_path, error_patterns)
        if rc is not None:
            return {
                "status": "finished" if rc == 0 else "failed",
                "return_code": rc,
                "reason": "process_exit",
                "error_excerpt": error_excerpt,
            }
        if error_excerpt:
            stop_process_tree(proc)
            return {
                "status": "killed",
                "return_code": proc.poll(),
                "reason": "error_pattern",
                "error_excerpt": error_excerpt,
            }

        if live_plateau_enabled:
            accs = parse_recent_valid_acc(log_path, keep_last=max(live_plateau_window + 2, 8))
            if len(accs) >= live_plateau_window and is_plateau_relative(accs, live_plateau_window, live_plateau_epsilon):
                stop_process_tree(proc)
                return {
                    "status": "killed",
                    "return_code": proc.poll(),
                    "reason": "plateau_accuracy",
                    "error_excerpt": f"live valid_acc plateau: recent={accs[-live_plateau_window:]}",
                    "latest_metric": accs[-1],
                }

        if log_path.exists():
            current_mtime = log_path.stat().st_mtime
            if current_mtime > last_mtime:
                last_mtime = current_mtime
        if time.time() - last_mtime > stall_timeout_sec:
            stop_process_tree(proc)
            return {
                "status": "killed",
                "return_code": proc.poll(),
                "reason": "stall_timeout",
                "error_excerpt": tail_text(log_path, max_lines=80),
            }
        if hard_timeout_sec and time.time() - start_ts > hard_timeout_sec:
            stop_process_tree(proc)
            return {
                "status": "killed",
                "return_code": proc.poll(),
                "reason": "hard_timeout",
                "error_excerpt": tail_text(log_path, max_lines=80),
            }
        time.sleep(poll_interval_sec)


def build_repair_instruction(exp_name: str, attempt: int, original_instruction: str, failure_excerpt: str) -> str:
    parts = [
        f"Repair experiment {exp_name} after failed training attempt {attempt}.",
        "Fix the code so training can continue without changing external interfaces or config keys.",
    ]
    if original_instruction:
        parts.extend(["", "Original experiment intent:", original_instruction])
    if failure_excerpt:
        parts.extend(["", "Failure excerpt:", failure_excerpt])
    return "\n".join(parts).strip()


def is_plateau_absolute(metrics: List[float], window: int, epsilon: float) -> bool:
    if window <= 1 or len(metrics) < window:
        return False
    recent = metrics[-window:]
    return (max(recent) - min(recent)) <= epsilon


def is_plateau_relative(metrics: List[float], window: int, epsilon_ratio: float) -> bool:
    if window <= 1 or len(metrics) < window:
        return False
    recent = metrics[-window:]
    for prev, curr in zip(recent[:-1], recent[1:]):
        base = max(abs(prev), 1e-8)
        ratio = abs(curr - prev) / base
        if ratio >= epsilon_ratio:
            return False
    return True


def should_trigger_retrain_after_finish(
    current_metric: float,
    previous_metric: Optional[float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return {"triggered": False, "reason": "disabled"}
    if previous_metric is None:
        return {"triggered": False, "reason": "no_previous_metric"}
    ratio_threshold = float(cfg.get("change_ratio_threshold", 0.01))
    poor_threshold = float(cfg.get("poor_metric_threshold", 0.78))
    change_ratio = abs(current_metric - previous_metric) / max(abs(previous_metric), 1e-8)
    triggered = change_ratio < ratio_threshold and current_metric < poor_threshold
    return {
        "triggered": triggered,
        "reason": "ratio_below_threshold_and_metric_poor" if triggered else "no_trigger",
        "change_ratio": change_ratio,
        "change_ratio_threshold": ratio_threshold,
        "poor_metric_threshold": poor_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent controller: watch training, repair code, retry, and promote better models.")
    parser.add_argument("--strategy", required=True, help="Controller strategy json")
    args = parser.parse_args()

    strategy_path = Path(args.strategy).resolve()
    strategy = load_json(strategy_path)
    project_root = Path(strategy.get("project_root", ".")).resolve()
    plan_path = (project_root / strategy["base_plan"]).resolve()
    plan = load_json(plan_path)

    base_config_path = (project_root / plan["base_config"]).resolve()
    base_cfg = load_json(base_config_path)
    python_bin = plan.get("python_bin", sys.executable)
    train_script = plan.get("train_script", "train.py")
    exp_root = (project_root / "exp").resolve()
    agent_cfg = plan.get("agent", {})
    agent_tasks = {task["name"]: task for task in plan.get("agent_tasks", [])}

    work_root = (project_root / strategy.get("work_dir", "auto_runs/background_controller")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    events_path = work_root / "events.jsonl"
    state_path = work_root / "controller_state.json"
    results_path = work_root / "results.json"
    snapshots_root = work_root / "snapshots"
    pidfile = work_root / "controller.pid"

    poll_interval_sec = int(strategy.get("poll_interval_sec", 20))
    stall_timeout_sec = int(strategy.get("stall_timeout_sec", 900))
    hard_timeout_sec = strategy.get("hard_timeout_sec")
    hard_timeout_sec = None if hard_timeout_sec is None else int(hard_timeout_sec)
    max_attempts = int(strategy.get("max_attempts_per_experiment", 2))
    min_improvement = float(strategy.get("min_improvement", 0.001))
    target_metric = str(strategy.get("target_metric", "valid_f1_macro"))
    target_valid_f1 = strategy.get("target_valid_f1")
    target_valid_f1 = None if target_valid_f1 is None else float(target_valid_f1)
    error_patterns = compile_patterns(strategy.get("error_patterns", DEFAULT_ERROR_PATTERNS))
    fallback_targets = strategy.get("repair", {}).get("fallback_target_files", ["model/DynRT.py"])
    managed_files = collect_managed_files(plan, fallback_targets)
    plateau_cfg = strategy.get("plateau", {})
    plateau_enabled = bool(plateau_cfg.get("enabled", False))
    plateau_window = int(plateau_cfg.get("window", 3))
    plateau_epsilon = float(plateau_cfg.get("epsilon", 0.001))
    plateau_mode = str(plateau_cfg.get("mode", "absolute")).lower()
    plateau_poor_threshold = float(plateau_cfg.get("poor_metric_threshold", 0.75))
    plateau_target_files = plateau_cfg.get("target_files", fallback_targets)
    plateau_instruction_template = plateau_cfg.get(
        "instruction",
        (
            "Model performance is in a plateau with tiny fluctuations and still below target. "
            "Refactor the model architecture for stronger multimodal interaction and better generalization, "
            "while keeping external interfaces and tensor contract unchanged."
        ),
    )
    live_plateau_enabled = bool(plateau_cfg.get("live_enabled", True))
    post_train_retrain_cfg = strategy.get("post_train_retrain", {})
    retention_cfg = strategy.get("retention", {})
    retention_enabled = bool(retention_cfg.get("enabled", True))
    retention_keep_exp_last = int(retention_cfg.get("keep_exp_last", 4))
    retention_prune_checkpoints = bool(retention_cfg.get("prune_checkpoints", True))
    retention_keep_checkpoint_names = list(retention_cfg.get("keep_checkpoint_names", ["model_best.pth.tar", "new_model_best.pth.tar"]))
    retention_keep_attempt_dirs = int(retention_cfg.get("keep_attempt_dirs", 12))
    retention_keep_train_log_lines = int(retention_cfg.get("keep_train_log_lines", 1200))
    retention_keep_agent_log_lines = int(retention_cfg.get("keep_agent_log_lines", 400))
    retention_keep_events_lines = int(retention_cfg.get("keep_events_lines", 4000))
    retention_keep_results_last = int(retention_cfg.get("keep_results_last", 800))
    retention_drop_instruction_files = bool(retention_cfg.get("drop_instruction_files", False))
    retention_drop_agent_logs = bool(retention_cfg.get("drop_agent_logs", False))
    global_guidance = str(strategy.get("agent_global_guidance", "")).strip()
    guidance_file = str(strategy.get("agent_guidance_file", "")).strip()
    if guidance_file:
        gpath = (project_root / guidance_file).resolve()
        if gpath.exists():
            global_guidance = gpath.read_text(encoding="utf-8", errors="ignore").strip()
    skill_files = [str(x).strip() for x in strategy.get("agent_skill_files", []) if str(x).strip()]
    skill_max_chars = int(strategy.get("agent_skill_max_chars", 12000))
    skill_guidance, loaded_skill_paths = load_skill_guidance(project_root, skill_files, skill_max_chars)
    if skill_guidance:
        if global_guidance:
            global_guidance = f"{global_guidance}\n\nAutoresearch Skill Guidance:\n{skill_guidance}".strip()
        else:
            global_guidance = skill_guidance

    accepted_snapshot = snapshots_root / "accepted"
    save_snapshot(project_root, managed_files, accepted_snapshot)

    if results_path.exists():
        try:
            loaded = load_json(results_path)
            results: List[Dict[str, Any]] = loaded if isinstance(loaded, list) else []
        except Exception:
            results = []
    else:
        results = []
    best_metric: Optional[float] = None
    best_result: Optional[Dict[str, Any]] = None
    finished_metrics: List[float] = [float(x["metric_value"]) for x in results if x.get("metric_value") is not None]
    completed_names = {x.get("name") for x in results if x.get("train_status") == "finished" and x.get("metric_value") is not None}
    accepted_rows = [x for x in results if x.get("accepted") and x.get("metric_value") is not None]
    if accepted_rows:
        best_result = max(accepted_rows, key=lambda x: float(x["metric_value"]))
        best_metric = float(best_result["metric_value"])
    elif finished_metrics:
        best_metric = max(finished_metrics)
        for x in results:
            if x.get("metric_value") is not None and float(x["metric_value"]) == best_metric:
                best_result = x
                break
    plateau_rethink_pending = False

    run_root = (project_root / plan.get("output_dir", "auto_runs/background_controller_managed")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Ensure singleton controller and clean orphan worker from previous crashed sessions.
    prev_controller_pid = read_pidfile(pidfile)
    if prev_controller_pid and prev_controller_pid != os.getpid() and pid_alive(prev_controller_pid):
        kill_pid_safely(prev_controller_pid)
    if state_path.exists():
        try:
            state_payload = load_json(state_path)
            prev_train_pid = int(state_payload.get("pid") or 0)
            if prev_train_pid and pid_alive(prev_train_pid):
                kill_pid_safely(prev_train_pid)
        except Exception:
            pass
    write_pidfile(pidfile, os.getpid())

    write_event(events_path, {"event": "controller_start", "strategy": str(strategy_path), "run_root": str(run_root)})
    if loaded_skill_paths:
        write_event(events_path, {"event": "agent_skills_loaded", "paths": loaded_skill_paths})

    for idx, exp_cfg in enumerate(plan.get("experiments", []), start=1):
        exp_name = exp_cfg["name"]
        if exp_name in completed_names:
            write_event(events_path, {"event": "skip_completed_experiment", "name": exp_name})
            continue
        if target_valid_f1 is not None and best_metric is not None and best_metric >= target_valid_f1:
            write_event(events_path, {"event": "stop_target_reached", "best_metric": best_metric})
            break

        agent_task_name = exp_cfg.get("agent_task")
        original_instruction = ""
        target_files = list(fallback_targets)
        if agent_task_name and agent_task_name in agent_tasks:
            original_instruction = agent_tasks[agent_task_name].get("instruction", "")
            target_files = agent_tasks[agent_task_name].get("target_files", target_files)

        accepted_before_exp = snapshots_root / f"accepted_before_{idx:02d}_{exp_name}"
        if accepted_before_exp.exists():
            shutil.rmtree(accepted_before_exp)
        shutil.copytree(accepted_snapshot, accepted_before_exp)

        succeeded = False
        for attempt in range(1, max_attempts + 1):
            restore_snapshot(project_root, accepted_before_exp)

            exp_dir = run_root / f"{idx:02d}_{exp_name}_attempt{attempt}"
            if exp_dir.exists():
                shutil.rmtree(exp_dir, ignore_errors=True)
            exp_dir.mkdir(parents=True, exist_ok=True)
            write_event(events_path, {"event": "attempt_start", "name": exp_name, "attempt": attempt})

            if plateau_rethink_pending and attempt == 1:
                plateau_instruction = (
                    f"{plateau_instruction_template}\n\n"
                    f"Current best {target_metric}: {best_metric}\n"
                    f"Recent metrics: {finished_metrics[-plateau_window:]}"
                )
                rethink_name = f"plateau_rethink_{idx:02d}_{exp_name}"
                rc = run_agent_task(
                    workspace=project_root,
                    experiment_dir=exp_dir,
                    agent_cfg=agent_cfg,
                    task_name=rethink_name,
                    instruction=plateau_instruction,
                    target_files=plateau_target_files,
                    global_guidance=global_guidance,
                )
                write_event(
                    events_path,
                    {
                        "event": "plateau_rethink_applied",
                        "name": exp_name,
                        "attempt": attempt,
                        "return_code": rc,
                    },
                )
                if rc != 0:
                    write_event(events_path, {"event": "plateau_rethink_failed", "name": exp_name, "attempt": attempt})
                    continue
                plateau_rethink_pending = False

            if agent_task_name and attempt == 1:
                rc = run_agent_task(
                    workspace=project_root,
                    experiment_dir=exp_dir,
                    agent_cfg=agent_cfg,
                    task_name=agent_task_name,
                    instruction=original_instruction,
                    target_files=target_files,
                    global_guidance=global_guidance,
                )
                if rc != 0:
                    write_event(events_path, {"event": "agent_failed", "name": exp_name, "attempt": attempt, "return_code": rc})
                    continue

            if attempt > 1 and strategy.get("repair", {}).get("enabled", True):
                failure_excerpt = results[-1].get("failure_excerpt", "") if results else ""
                repair_name = f"repair_{exp_name}_{attempt}"
                repair_instruction = build_repair_instruction(exp_name, attempt, original_instruction, failure_excerpt)
                rc = run_agent_task(
                    workspace=project_root,
                    experiment_dir=exp_dir,
                    agent_cfg=agent_cfg,
                    task_name=repair_name,
                    instruction=repair_instruction,
                    target_files=target_files,
                    global_guidance=global_guidance,
                )
                write_event(events_path, {"event": "repair_applied", "name": exp_name, "attempt": attempt, "return_code": rc})
                if rc != 0:
                    continue

            config_payload = build_experiment_config(base_cfg, exp_cfg.get("config_overrides", {}), exp_name)
            config_file = exp_dir / "config.json"
            save_json(config_file, config_payload)
            kill_stale_train_by_config(config_file)

            before_dirs = list_exp_dirs(exp_root)
            cmd = [python_bin, train_script, str(config_file)]
            cmd.extend(exp_cfg.get("train_args", []))
            env = os.environ.copy()
            env.update({k: str(v) for k, v in exp_cfg.get("env", {}).items()})
            proc = launch_training(cmd, cwd=project_root, env=env, log_path=exp_dir / "train.log")

            state = {
                "status": "running",
                "experiment": exp_name,
                "attempt": attempt,
                "pid": proc.pid,
                "run_root": str(run_root),
                "work_root": str(work_root),
            }
            save_json(state_path, state)

            monitor = monitor_training(
                proc=proc,
                log_path=exp_dir / "train.log",
                error_patterns=error_patterns,
                poll_interval_sec=poll_interval_sec,
                stall_timeout_sec=stall_timeout_sec,
                hard_timeout_sec=hard_timeout_sec,
                live_plateau_enabled=live_plateau_enabled,
                live_plateau_window=plateau_window,
                live_plateau_epsilon=plateau_epsilon,
            )
            after_dirs = list_exp_dirs(exp_root)
            created_exp_dir = find_new_exp_dir(before_dirs, after_dirs)

            row: Dict[str, Any] = {
                "name": exp_name,
                "attempt": attempt,
                "agent_task": agent_task_name,
                "train_status": monitor["status"],
                "train_reason": monitor["reason"],
                "train_return_code": monitor.get("return_code"),
                "failure_excerpt": monitor.get("error_excerpt"),
                "run_dir": rel_to(exp_dir, project_root),
                "exp_dir": rel_to(created_exp_dir, project_root) if created_exp_dir else None,
            }
            if created_exp_dir:
                row.update(parse_structured_metrics(created_exp_dir))
            row["metric_value"] = monitor.get("latest_metric")
            if row["metric_value"] is None:
                row["metric_value"] = pick_metric(row, target_metric)
            results.append(row)
            if retention_enabled and retention_keep_results_last > 0 and len(results) > retention_keep_results_last:
                results = results[-retention_keep_results_last:]
            save_json(results_path, results)

            write_event(
                events_path,
                {
                    "event": "attempt_finish",
                    "name": exp_name,
                    "attempt": attempt,
                    "status": row["train_status"],
                    "metric_value": row["metric_value"],
                },
            )

            if row["train_reason"] == "plateau_accuracy":
                plateau_rethink_pending = True
                write_event(
                    events_path,
                    {
                        "event": "live_plateau_triggered",
                        "name": exp_name,
                        "attempt": attempt,
                        "latest_metric": row["metric_value"],
                        "epsilon": plateau_epsilon,
                    },
                )
                if attempt < max_attempts:
                    if retention_enabled:
                        try:
                            cleanup_attempt_artifacts(
                                attempt_dir=exp_dir,
                                keep_train_log_lines=retention_keep_train_log_lines,
                                keep_agent_log_lines=retention_keep_agent_log_lines,
                                drop_instruction_files=retention_drop_instruction_files,
                                drop_agent_logs=retention_drop_agent_logs,
                            )
                            pinned_exp_dirs: List[str] = []
                            if created_exp_dir:
                                pinned_exp_dirs.append(created_exp_dir.name)
                            if best_result and best_result.get("exp_dir"):
                                pinned_exp_dirs.append(Path(str(best_result["exp_dir"])).name)
                            prune_exp_history(
                                exp_root=exp_root,
                                keep_last=retention_keep_exp_last,
                                pinned_dirs=pinned_exp_dirs,
                                prune_checkpoints=retention_prune_checkpoints,
                                keep_checkpoint_names=retention_keep_checkpoint_names,
                            )
                            prune_run_attempt_history(run_root=run_root, keep_last=retention_keep_attempt_dirs, pinned_dirs=[exp_dir.name])
                            trim_text_file(events_path, retention_keep_events_lines)
                        except Exception as e:
                            write_event(events_path, {"event": "retention_error", "name": exp_name, "attempt": attempt, "error": str(e)})
                    continue

            if row["train_status"] == "finished" and row["metric_value"] is not None:
                previous_metric = finished_metrics[-1] if finished_metrics else None
                finished_metrics.append(float(row["metric_value"]))
                improved = best_metric is None or (row["metric_value"] - best_metric) >= min_improvement
                if improved:
                    save_snapshot(project_root, managed_files, accepted_snapshot)
                    best_metric = row["metric_value"]
                    best_result = row
                    row["accepted"] = True
                    write_event(events_path, {"event": "accepted", "name": exp_name, "attempt": attempt, "metric_value": best_metric})
                else:
                    restore_snapshot(project_root, accepted_before_exp)
                    row["accepted"] = False
                    write_event(events_path, {"event": "reverted_no_improve", "name": exp_name, "attempt": attempt, "metric_value": row["metric_value"], "best_metric": best_metric})

                if plateau_enabled:
                    if plateau_mode == "relative":
                        plateau_hit = is_plateau_relative(finished_metrics, plateau_window, plateau_epsilon)
                    else:
                        plateau_hit = is_plateau_absolute(finished_metrics, plateau_window, plateau_epsilon)
                else:
                    plateau_hit = False
                row["plateau_detected"] = plateau_hit
                if plateau_hit and (best_metric is None or best_metric < plateau_poor_threshold):
                    plateau_rethink_pending = True
                    write_event(
                        events_path,
                        {
                            "event": "plateau_detected",
                            "name": exp_name,
                            "attempt": attempt,
                            "best_metric": best_metric,
                            "recent_metrics": finished_metrics[-plateau_window:],
                            "plateau_mode": plateau_mode,
                            "epsilon": plateau_epsilon,
                            "poor_metric_threshold": plateau_poor_threshold,
                        },
                    )
                retrain_decision = should_trigger_retrain_after_finish(
                    current_metric=float(row["metric_value"]),
                    previous_metric=previous_metric,
                    cfg=post_train_retrain_cfg,
                )
                row["previous_metric"] = previous_metric
                row["retrain_change_ratio"] = retrain_decision.get("change_ratio")
                row["retrain_triggered"] = retrain_decision["triggered"]
                row["retrain_reason"] = retrain_decision["reason"]
                if retrain_decision["triggered"]:
                    trigger_record = {
                        "name": exp_name,
                        "attempt": attempt,
                        "current_metric": row["metric_value"],
                        "previous_metric": previous_metric,
                        "change_ratio": retrain_decision.get("change_ratio"),
                        "change_ratio_threshold": retrain_decision.get("change_ratio_threshold"),
                        "poor_metric_threshold": retrain_decision.get("poor_metric_threshold"),
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                    save_json(exp_dir / "pre_retrain_record.json", trigger_record)
                    write_event(events_path, {"event": "post_train_retrain_triggered", **trigger_record})
                    plateau_rethink_pending = True
                save_json(results_path, results)
                if retrain_decision["triggered"] and attempt < max_attempts:
                    if retention_enabled:
                        try:
                            cleanup_attempt_artifacts(
                                attempt_dir=exp_dir,
                                keep_train_log_lines=retention_keep_train_log_lines,
                                keep_agent_log_lines=retention_keep_agent_log_lines,
                                drop_instruction_files=retention_drop_instruction_files,
                                drop_agent_logs=retention_drop_agent_logs,
                            )
                            pinned_exp_dirs: List[str] = []
                            if created_exp_dir:
                                pinned_exp_dirs.append(created_exp_dir.name)
                            if best_result and best_result.get("exp_dir"):
                                pinned_exp_dirs.append(Path(str(best_result["exp_dir"])).name)
                            prune_exp_history(
                                exp_root=exp_root,
                                keep_last=retention_keep_exp_last,
                                pinned_dirs=pinned_exp_dirs,
                                prune_checkpoints=retention_prune_checkpoints,
                                keep_checkpoint_names=retention_keep_checkpoint_names,
                            )
                            prune_run_attempt_history(run_root=run_root, keep_last=retention_keep_attempt_dirs, pinned_dirs=[exp_dir.name])
                            trim_text_file(events_path, retention_keep_events_lines)
                        except Exception as e:
                            write_event(events_path, {"event": "retention_error", "name": exp_name, "attempt": attempt, "error": str(e)})
                    continue
                if retention_enabled:
                    try:
                        cleanup_attempt_artifacts(
                            attempt_dir=exp_dir,
                            keep_train_log_lines=retention_keep_train_log_lines,
                            keep_agent_log_lines=retention_keep_agent_log_lines,
                            drop_instruction_files=retention_drop_instruction_files,
                            drop_agent_logs=retention_drop_agent_logs,
                        )
                        pinned_exp_dirs: List[str] = []
                        if created_exp_dir:
                            pinned_exp_dirs.append(created_exp_dir.name)
                        if best_result and best_result.get("exp_dir"):
                            pinned_exp_dirs.append(Path(str(best_result["exp_dir"])).name)
                        prune_exp_history(
                            exp_root=exp_root,
                            keep_last=retention_keep_exp_last,
                            pinned_dirs=pinned_exp_dirs,
                            prune_checkpoints=retention_prune_checkpoints,
                            keep_checkpoint_names=retention_keep_checkpoint_names,
                        )
                        prune_run_attempt_history(run_root=run_root, keep_last=retention_keep_attempt_dirs, pinned_dirs=[exp_dir.name])
                        trim_text_file(events_path, retention_keep_events_lines)
                    except Exception as e:
                        write_event(events_path, {"event": "retention_error", "name": exp_name, "attempt": attempt, "error": str(e)})
                succeeded = True
                break

            if retention_enabled:
                try:
                    cleanup_attempt_artifacts(
                        attempt_dir=exp_dir,
                        keep_train_log_lines=retention_keep_train_log_lines,
                        keep_agent_log_lines=retention_keep_agent_log_lines,
                        drop_instruction_files=retention_drop_instruction_files,
                        drop_agent_logs=retention_drop_agent_logs,
                    )
                    pinned_exp_dirs: List[str] = []
                    if created_exp_dir:
                        pinned_exp_dirs.append(created_exp_dir.name)
                    if best_result and best_result.get("exp_dir"):
                        pinned_exp_dirs.append(Path(str(best_result["exp_dir"])).name)
                    prune_exp_history(
                        exp_root=exp_root,
                        keep_last=retention_keep_exp_last,
                        pinned_dirs=pinned_exp_dirs,
                        prune_checkpoints=retention_prune_checkpoints,
                        keep_checkpoint_names=retention_keep_checkpoint_names,
                    )
                    prune_run_attempt_history(run_root=run_root, keep_last=retention_keep_attempt_dirs, pinned_dirs=[exp_dir.name])
                    trim_text_file(events_path, retention_keep_events_lines)
                except Exception as e:
                    write_event(events_path, {"event": "retention_error", "name": exp_name, "attempt": attempt, "error": str(e)})

        if not succeeded:
            restore_snapshot(project_root, accepted_before_exp)
            write_event(events_path, {"event": "experiment_abandoned", "name": exp_name})
            save_json(results_path, results)

    final_state = {
        "status": "completed",
        "best_metric": best_metric,
        "best_result": best_result,
        "run_root": str(run_root),
        "work_root": str(work_root),
    }
    save_json(state_path, final_state)
    write_event(events_path, {"event": "controller_finish", "best_metric": best_metric, "best_result": best_result})


if __name__ == "__main__":
    main()
