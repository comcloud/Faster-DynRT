#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(project_root: Path, plan_path: Path, dry_run: bool) -> Dict[str, Any]:
    cmd = ["python", "automation/auto_agent_experiment.py", "--plan", str(plan_path)]
    if dry_run:
        cmd.append("--dry-run")
    p = subprocess.run(cmd, cwd=str(project_root), check=False)
    if p.returncode != 0:
        raise RuntimeError(f"auto_agent_experiment failed: code={p.returncode}")
    run_root = load_json(plan_path).get("output_dir")
    summary = load_json((project_root / run_root / "summary.json").resolve())
    if not summary:
        raise RuntimeError("Empty summary from auto experiment")
    return summary[-1]


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def pick_valid_metric(row: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["valid_f1_macro", "best_metric_value", "valid_f1", "valid_f1_micro", "valid_acc"]:
        v = as_float(row.get(key))
        if v is not None:
            return {"metric_key": key, "metric_value": v}
    return {"metric_key": None, "metric_value": None}


def mk_single_plan(base_plan: Dict[str, Any], exp: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    payload = copy.deepcopy(base_plan)
    payload["output_dir"] = output_dir
    payload["experiments"] = [exp]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-improving runner: auto switch model/params by metric.")
    parser.add_argument("--strategy", required=True, help="Path to strategy json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategy_path = Path(args.strategy).resolve()
    strategy = load_json(strategy_path)
    project_root = Path(strategy.get("project_root", ".")).resolve()
    base_plan = load_json((project_root / strategy["base_plan"]).resolve())

    max_rounds = int(strategy.get("max_rounds", 10))
    min_improve = float(strategy.get("min_improvement", 0.001))
    target_valid_f1 = strategy.get("target_valid_f1")
    target_valid_f1 = None if target_valid_f1 is None else float(target_valid_f1)

    model_candidates: List[Dict[str, Any]] = strategy.get("model_candidates", [])
    param_candidates: List[Dict[str, Any]] = strategy.get("param_candidates", [])
    baseline_exp: Dict[str, Any] = strategy.get("baseline_experiment", {"name": "baseline", "config_overrides": {"opt.seed": 2}})

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_root = (project_root / strategy.get("work_dir", f"auto_runs/self_improve_{stamp}")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_valid = -1.0
    best_candidate: Dict[str, Any] = {"round": None, "name": None, "metric_key": None, "metric_value": None}
    round_idx = 1

    # round 1: baseline
    plan_path = work_root / f"round_{round_idx:02d}_baseline.plan.json"
    single = mk_single_plan(base_plan, baseline_exp, output_dir=f"{work_root.relative_to(project_root)}/round_{round_idx:02d}_baseline")
    save_json(plan_path, single)
    res = run_once(project_root, plan_path, dry_run=args.dry_run)
    picked = pick_valid_metric(res)
    v = picked["metric_value"]
    best_valid = v if v is not None else best_valid
    best_candidate = {
        "round": round_idx,
        "name": res.get("name"),
        "metric_key": picked["metric_key"],
        "metric_value": v,
    }
    history.append(
        {
            "round": round_idx,
            "type": "baseline",
            "result": res,
            "picked_metric_key": picked["metric_key"],
            "picked_metric_value": v,
            "improved": True if v is not None else False,
            "decision_reason": "baseline initialization",
            "best_valid_f1": best_valid,
            "best_candidate": best_candidate,
        }
    )
    round_idx += 1

    model_cursor = 0
    param_cursor = 0
    stagnant = 0

    while round_idx <= max_rounds:
        if target_valid_f1 is not None and best_valid >= target_valid_f1:
            break

        exp: Optional[Dict[str, Any]] = None
        exp_type = ""
        if model_cursor < len(model_candidates):
            cand = model_candidates[model_cursor]
            exp = {
                "name": cand.get("name", f"model_{model_cursor+1}"),
                "agent_task": cand["agent_task"],
                "config_overrides": cand.get("config_overrides", {"opt.seed": 2}),
                "env": cand.get("env", {"CUDA_VISIBLE_DEVICES": "0"}),
            }
            model_cursor += 1
            exp_type = "model"
        elif param_cursor < len(param_candidates):
            cand = param_candidates[param_cursor]
            exp = {
                "name": cand.get("name", f"param_{param_cursor+1}"),
                "config_overrides": cand.get("config_overrides", {"opt.seed": 2}),
                "env": cand.get("env", {"CUDA_VISIBLE_DEVICES": "0"}),
            }
            if "agent_task" in cand:
                exp["agent_task"] = cand["agent_task"]
            param_cursor += 1
            exp_type = "param"
        else:
            break

        plan_path = work_root / f"round_{round_idx:02d}_{exp_type}.plan.json"
        single = mk_single_plan(base_plan, exp, output_dir=f"{work_root.relative_to(project_root)}/round_{round_idx:02d}_{exp_type}")
        save_json(plan_path, single)
        res = run_once(project_root, plan_path, dry_run=args.dry_run)
        picked = pick_valid_metric(res)
        v = picked["metric_value"]

        prev_best = best_valid
        improved = v is not None and (v - prev_best) >= min_improve
        replaced_best = False
        if improved:
            best_valid = v  # type: ignore[assignment]
            stagnant = 0
            replaced_best = True
            best_candidate = {
                "round": round_idx,
                "name": res.get("name"),
                "metric_key": picked["metric_key"],
                "metric_value": v,
            }
        else:
            stagnant += 1

        history.append(
            {
                "round": round_idx,
                "type": exp_type,
                "experiment": exp,
                "result": res,
                "picked_metric_key": picked["metric_key"],
                "picked_metric_value": v,
                "improved": improved,
                "replaced_best": replaced_best,
                "decision_reason": (
                    f"improved by {v - prev_best:+.6f} >= min_improvement {min_improve}"
                    if improved and v is not None
                    else (
                        f"no valid metric parsed"
                        if v is None
                        else f"delta {v - prev_best:+.6f} < min_improvement {min_improve}"
                    )
                ),
                "best_valid_f1": best_valid,
                "stagnant": stagnant,
                "best_candidate": best_candidate,
            }
        )

        round_idx += 1

    out = work_root / "self_improve_summary.json"
    save_json(out, {"best_valid_f1": best_valid, "best_candidate": best_candidate, "history": history})

    md_lines = [
        "# Self Improve Decision Trace",
        "",
        f"- best_valid_f1: {best_valid}",
        f"- best_candidate: {best_candidate}",
        "",
        "| round | type | exp_name | picked_metric | improved | replaced_best | reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in history:
        result = item.get("result", {})
        metric_key = item.get("picked_metric_key")
        metric_val = item.get("picked_metric_value")
        picked = "" if metric_key is None else f"{metric_key}={metric_val}"
        md_lines.append(
            "| {round} | {type} | {name} | {picked} | {improved} | {replaced_best} | {reason} |".format(
                round=item.get("round", ""),
                type=item.get("type", ""),
                name=result.get("name", ""),
                picked=picked,
                improved=item.get("improved", ""),
                replaced_best=item.get("replaced_best", ""),
                reason=item.get("decision_reason", ""),
            )
        )
    trace_md = work_root / "decision_trace.md"
    trace_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"done: {out}")
    print(f"trace: {trace_md}")


if __name__ == "__main__":
    main()
