#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path, tail_n: int = 30) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows[-tail_n:]


def main() -> None:
    root = Path(__file__).resolve().parent
    report = read_json(root / "data_report.remote.json")
    state = read_json(root / "controller_state.remote.json")
    events = read_events(root / "events.remote.jsonl", tail_n=40)

    payload = {
        "report": report,
        "state": state,
        "events": events,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DynRT Remote Data Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #64748b;
      --accent: #0f766e;
      --line: #e2e8f0;
    }}
    body {{
      margin: 0;
      padding: 20px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    .wrap {{
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }}
    .title {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }}
    .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }}
    .v {{
      font-size: 18px;
      font-weight: 600;
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 6px;
      word-break: break-all;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      white-space: nowrap;
    }}
    .muted {{
      color: var(--muted);
    }}
    .ok {{
      color: var(--accent);
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1 class="title">DynRT 远程数据与训练看板</h1>
      <div class="muted">本页为本地静态快照。刷新方式：重新拉取远程 JSON 后重新运行构建脚本。</div>
    </div>

    <div class="card">
      <div class="grid" id="stats"></div>
    </div>

    <div class="card">
      <h2 class="title">关键路径</h2>
      <table id="paths"></table>
    </div>

    <div class="card">
      <h2 class="title">控制器状态</h2>
      <table id="state"></table>
    </div>

    <div class="card">
      <h2 class="title">最近事件</h2>
      <table id="events"></table>
    </div>
  </div>
  <script>
    const data = {payload_json};
    const report = data.report || {{}};
    const state = data.state || {{}};
    const events = data.events || [];

    const counts = (report.counts || {{}});
    const splits = (report.prepared_aligned || {{}});
    const stats = [
      ["图片文件数", counts.images_files ?? "-"],
      ["npy 文件数", counts.tensor_npy_files ?? "-"],
      ["压缩包数量", counts.compressed_files_under_dataset ?? "-"],
      ["train ids", splits.train_ids ?? "-"],
      ["valid ids", splits.valid_ids ?? "-"],
      ["test ids", splits.test_ids ?? "-"],
      ["报告生成时间", report.generated_at ?? "-"],
      ["控制器状态", state.status ?? "-"],
    ];
    const statsEl = document.getElementById("stats");
    stats.forEach(([k, v]) => {{
      const div = document.createElement("div");
      div.innerHTML = `<div class="k">${{k}}</div><div class="v">${{v}}</div>`;
      statsEl.appendChild(div);
    }});

    function renderKVTable(elId, obj) {{
      const table = document.getElementById(elId);
      const header = document.createElement("tr");
      header.innerHTML = "<th>字段</th><th>值</th>";
      table.appendChild(header);
      Object.entries(obj || {{}}).forEach(([k, v]) => {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${{k}}</td><td><code>${{String(v)}}</code></td>`;
        table.appendChild(tr);
      }});
    }}
    renderKVTable("paths", report.paths || {{}});
    renderKVTable("state", state || {{}});

    const eventsTable = document.getElementById("events");
    const h = document.createElement("tr");
    h.innerHTML = "<th>timestamp</th><th>event</th><th>detail</th>";
    eventsTable.appendChild(h);
    events.slice().reverse().forEach((ev) => {{
      const row = document.createElement("tr");
      const detail = Object.assign({{}}, ev);
      delete detail.timestamp;
      delete detail.event;
      row.innerHTML = `<td>${{ev.timestamp || "-"}}</td><td class="ok">${{ev.event || "-"}}</td><td><code>${{JSON.stringify(detail)}}</code></td>`;
      eventsTable.appendChild(row);
    }});
  </script>
</body>
</html>
"""
    out = root / "remote_data_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
