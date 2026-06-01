#!/usr/bin/env python3
"""Generate an interactive token-usage heatmap for Claude Code sessions.

Claude Code records every session as a JSONL transcript under
``~/.claude/projects/<encoded-project>/<session>.jsonl``. Each ``assistant``
message carries a ``timestamp`` and a ``message.usage`` block with input,
output, and cache token counts. This script aggregates those counts and emits a
single self-contained HTML page (no external assets, no network) with three
views:

1. Weekday x Hour  -- a classic 7x24 grid of when tokens are spent.
2. Project x Date  -- one row per project, columns per day.
3. Calendar        -- a GitHub-contributions-style daily grid.

All token kinds are summed by default:
``input + output + cache_creation + cache_read``.

Usage:
    python scripts/cc_token_heatmap.py [--dir DIR] [--out FILE] [--open]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def default_projects_dir() -> Path:
    return Path(os.path.expanduser("~/.claude/projects"))


def total_tokens(usage: dict) -> int:
    """Sum every token kind reported in a usage block."""
    if not isinstance(usage, dict):
        return 0
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("output_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
    )


def project_label(record: dict, jsonl_path: Path) -> str:
    """Human-friendly project name: basename of cwd, else the project dir."""
    cwd = record.get("cwd")
    if isinstance(cwd, str) and cwd:
        return os.path.basename(cwd.rstrip("/")) or cwd
    # Fall back to the encoded project directory name.
    return jsonl_path.parent.name.lstrip("-").replace("-", "/")


def iter_usage_events(projects_dir: Path):
    """Yield (datetime_local, project, tokens, kinds_dict) for each event."""
    files = sorted(projects_dir.rglob("*.jsonl"))
    for path in files:
        try:
            fh = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message") or {}
                usage = msg.get("usage")
                tokens = total_tokens(usage)
                if tokens <= 0:
                    continue
                ts = rec.get("timestamp")
                if not ts:
                    continue
                try:
                    # Timestamps are ISO-8601 UTC ("...Z").
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    dt = dt.astimezone()  # convert to local time
                except ValueError:
                    continue
                kinds = {
                    "input": int((usage or {}).get("input_tokens") or 0),
                    "output": int((usage or {}).get("output_tokens") or 0),
                    "cache_creation": int(
                        (usage or {}).get("cache_creation_input_tokens") or 0
                    ),
                    "cache_read": int(
                        (usage or {}).get("cache_read_input_tokens") or 0
                    ),
                }
                yield dt, project_label(rec, path), tokens, kinds


def aggregate(projects_dir: Path) -> dict:
    weekday_hour = defaultdict(int)          # (weekday 0=Mon, hour) -> tokens
    project_day = defaultdict(int)           # (project, "YYYY-MM-DD") -> tokens
    day_totals = defaultdict(int)            # "YYYY-MM-DD" -> tokens
    project_totals = defaultdict(int)        # project -> tokens
    kind_totals = defaultdict(int)           # kind -> tokens
    n_events = 0
    grand_total = 0

    for dt, project, tokens, kinds in iter_usage_events(projects_dir):
        n_events += 1
        grand_total += tokens
        day = dt.strftime("%Y-%m-%d")
        weekday_hour[(dt.weekday(), dt.hour)] += tokens
        project_day[(project, day)] += tokens
        day_totals[day] += tokens
        project_totals[project] += tokens
        for k, v in kinds.items():
            kind_totals[k] += v

    return {
        "weekday_hour": {f"{w},{h}": t for (w, h), t in weekday_hour.items()},
        "project_day": {f"{p}\t{d}": t for (p, d), t in project_day.items()},
        "day_totals": dict(day_totals),
        "project_totals": dict(project_totals),
        "kind_totals": dict(kind_totals),
        "n_events": n_events,
        "grand_total": grand_total,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Code Token Heatmap</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #e6edf3; --muted: #8b949e;
    --grid: #30363d; --accent: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  header { padding: 24px 28px 8px; }
  h1 { margin: 0 0 4px; font-size: 20px; }
  .sub { color: var(--muted); font-size: 13px; }
  .stats { display: flex; flex-wrap: wrap; gap: 16px; padding: 12px 28px 4px; }
  .stat { background: var(--panel); border: 1px solid var(--grid); border-radius: 8px;
    padding: 10px 14px; min-width: 120px; }
  .stat .v { font-size: 18px; font-weight: 600; }
  .stat .k { color: var(--muted); font-size: 12px; }
  section { padding: 20px 28px; }
  h2 { font-size: 15px; margin: 0 0 12px; color: var(--fg); }
  .grid { display: grid; gap: 3px; }
  .cell { width: 100%; aspect-ratio: 1 / 1; border-radius: 3px; background: #1c2128;
    min-width: 12px; }
  .label { color: var(--muted); font-size: 11px; text-align: center; }
  .rowlabel { color: var(--muted); font-size: 11px; display: flex; align-items: center;
    justify-content: flex-end; padding-right: 6px; white-space: nowrap; }
  .legend { display: flex; align-items: center; gap: 6px; color: var(--muted);
    font-size: 12px; margin-top: 10px; }
  .legend .cell { width: 14px; aspect-ratio: 1; }
  #tip { position: fixed; pointer-events: none; background: #000; color: #fff;
    border: 1px solid var(--grid); border-radius: 6px; padding: 6px 9px; font-size: 12px;
    opacity: 0; transition: opacity .08s; z-index: 10; white-space: nowrap; }
  .cal { display: grid; grid-auto-flow: column; grid-template-rows: repeat(7, 13px);
    gap: 3px; overflow-x: auto; padding-bottom: 6px; }
  .cal .cell { width: 13px; height: 13px; aspect-ratio: auto; }
  table { border-collapse: collapse; }
  .empty { color: var(--muted); padding: 12px 0; }
</style>
</head>
<body>
<header>
  <h1>Claude Code &mdash; Token Usage Heatmap</h1>
  <div class="sub">Generated __GENERATED_AT__ &middot; local time &middot; all token kinds summed</div>
</header>

<div class="stats" id="stats"></div>

<section>
  <h2>Weekday &times; Hour</h2>
  <div id="wh"></div>
  <div class="legend">Less
    <span class="cell" data-l="0"></span><span class="cell" data-l="1"></span>
    <span class="cell" data-l="2"></span><span class="cell" data-l="3"></span>
    <span class="cell" data-l="4"></span> More
  </div>
</section>

<section>
  <h2>Project &times; Date</h2>
  <div id="pd"></div>
</section>

<section>
  <h2>Daily calendar</h2>
  <div id="cal"></div>
</section>

<div id="tip"></div>

<script>
const DATA = __DATA__;

const tip = document.getElementById("tip");
function fmt(n){ return n.toLocaleString(); }
function showTip(e, html){
  tip.innerHTML = html; tip.style.opacity = 1;
  const pad = 12;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight) y = e.clientY - r.height - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip(){ tip.style.opacity = 0; }

// GitHub-style 5-step green scale.
const SCALE = ["#1c2128","#0e4429","#006d32","#26a641","#39d353"];
function level(v, max){
  if (v <= 0 || max <= 0) return 0;
  const r = v / max;                       // log-ish bucketing
  if (r > 0.6) return 4;
  if (r > 0.3) return 3;
  if (r > 0.12) return 2;
  return 1;
}
function color(v, max){ return SCALE[level(v, max)]; }

// Stat cards.
(function(){
  const s = document.getElementById("stats");
  const k = DATA.kind_totals || {};
  const cards = [
    ["Total tokens", fmt(DATA.grand_total)],
    ["Assistant turns", fmt(DATA.n_events)],
    ["Input", fmt(k.input||0)],
    ["Output", fmt(k.output||0)],
    ["Cache create", fmt(k.cache_creation||0)],
    ["Cache read", fmt(k.cache_read||0)],
  ];
  s.innerHTML = cards.map(([key,val]) =>
    `<div class="stat"><div class="v">${val}</div><div class="k">${key}</div></div>`).join("");
})();

// View 1: weekday x hour.
(function(){
  const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
  const wh = DATA.weekday_hour || {};
  let max = 0;
  for (const v of Object.values(wh)) max = Math.max(max, v);
  const root = document.getElementById("wh");
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.style.gridTemplateColumns = "40px repeat(24, 1fr)";
  // header row
  grid.appendChild(cellDiv("label",""));
  for (let h=0; h<24; h++){ const c = cellDiv("label", h%3===0 ? h : ""); grid.appendChild(c); }
  for (let d=0; d<7; d++){
    grid.appendChild(cellDiv("rowlabel", days[d]));
    for (let h=0; h<24; h++){
      const v = wh[d+","+h] || 0;
      const c = document.createElement("div");
      c.className = "cell"; c.style.background = color(v, max);
      c.addEventListener("mousemove", e =>
        showTip(e, `<b>${days[d]} ${String(h).padStart(2,"0")}:00</b><br>${fmt(v)} tokens`));
      c.addEventListener("mouseleave", hideTip);
      grid.appendChild(c);
    }
  }
  root.appendChild(grid);
  function cellDiv(cls, txt){ const e=document.createElement("div"); e.className=cls; e.textContent=txt; return e; }
})();

// View 2: project x date.
(function(){
  const pd = DATA.project_day || {};
  const projTotals = DATA.project_totals || {};
  const root = document.getElementById("pd");
  // collect projects (sorted by total desc) and the date range.
  const projects = Object.keys(projTotals).sort((a,b)=>projTotals[b]-projTotals[a]);
  const dateSet = new Set();
  for (const key of Object.keys(pd)) dateSet.add(key.split("\\t")[1]);
  const dates = Array.from(dateSet).sort();
  if (!projects.length || !dates.length){ root.innerHTML = '<div class="empty">No data.</div>'; return; }
  let max = 0;
  for (const v of Object.values(pd)) max = Math.max(max, v);
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.style.gridTemplateColumns = `160px repeat(${dates.length}, minmax(12px, 1fr))`;
  grid.appendChild(cellDiv("label",""));
  for (const d of dates){ grid.appendChild(cellDiv("label", d.slice(5))); }
  for (const p of projects){
    grid.appendChild(cellDiv("rowlabel", p));
    for (const d of dates){
      const v = pd[p+"\\t"+d] || 0;
      const c = document.createElement("div");
      c.className = "cell"; c.style.background = color(v, max);
      c.addEventListener("mousemove", e =>
        showTip(e, `<b>${p}</b><br>${d}<br>${fmt(v)} tokens`));
      c.addEventListener("mouseleave", hideTip);
      grid.appendChild(c);
    }
  }
  root.appendChild(grid);
  function cellDiv(cls, txt){ const e=document.createElement("div"); e.className=cls; e.textContent=txt; return e; }
})();

// View 3: calendar (GitHub-style).
(function(){
  const dt = DATA.day_totals || {};
  const dates = Object.keys(dt).sort();
  const root = document.getElementById("cal");
  if (!dates.length){ root.innerHTML = '<div class="empty">No data.</div>'; return; }
  let max = 0; for (const v of Object.values(dt)) max = Math.max(max, v);
  // build a continuous range from first to last day.
  const start = new Date(dates[0] + "T00:00:00");
  const end = new Date(dates[dates.length-1] + "T00:00:00");
  // pad start back to Monday for clean columns.
  const startDow = (start.getDay()+6)%7; // 0=Mon
  start.setDate(start.getDate() - startDow);
  const cal = document.createElement("div");
  cal.className = "cal";
  for (let d = new Date(start); d <= end; d.setDate(d.getDate()+1)){
    const key = d.toISOString().slice(0,10);
    const v = dt[key] || 0;
    const c = document.createElement("div");
    c.className = "cell"; c.style.background = color(v, max);
    c.addEventListener("mousemove", e =>
      showTip(e, `<b>${key}</b><br>${fmt(v)} tokens`));
    c.addEventListener("mouseleave", hideTip);
    cal.appendChild(c);
  }
  root.appendChild(cal);
})();
</script>
</body>
</html>
"""


def render_html(data: dict) -> str:
    return HTML_TEMPLATE.replace(
        "__DATA__", json.dumps(data, ensure_ascii=False)
    ).replace("__GENERATED_AT__", data.get("generated_at", ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=default_projects_dir(),
                        help="Claude Code projects dir (default: ~/.claude/projects)")
    parser.add_argument("--out", type=Path, default=Path("cc_token_heatmap.html"),
                        help="Output HTML file (default: cc_token_heatmap.html)")
    parser.add_argument("--open", action="store_true",
                        help="Open the generated file in the default browser")
    args = parser.parse_args(argv)

    if not args.dir.exists():
        print(f"error: projects dir not found: {args.dir}", file=sys.stderr)
        return 1

    data = aggregate(args.dir)
    if data["n_events"] == 0:
        print(f"warning: no usage events found under {args.dir}", file=sys.stderr)

    args.out.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {args.out}  ({data['n_events']} turns, "
          f"{data['grand_total']:,} tokens)")

    if args.open:
        import webbrowser
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
