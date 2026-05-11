"""
Report Agent — seventh and final node in the LangGraph pipeline.

Responsibilities:
  1. Consume the full WorkflowState (all prior agent outputs).
  2. Render a structured Markdown report covering every pipeline stage.
  3. Convert the Markdown to a self-contained HTML report.
  4. Save both files to reports/markdown/ and reports/html/.
  5. Write report_markdown_path and report_html_path back into WorkflowState.
  6. Append a NodeResult.

Report sections:
  1. Executive Summary
  2. Dataset Overview
  3. Data Quality (EDA findings)
  4. Cleaning Summary
  5. Feature Engineering Summary
  6. Model Training Results (per-model metrics table)
  7. Model Evaluation & Ranking (leaderboard)
  8. Best Model
  9. Pipeline Execution Timeline (node durations)
  10. Appendix (chart paths, artifact paths)

Design rules:
  - Pure function signature: (WorkflowState) → dict patch.
  - No database access.
  - No LangGraph imports.
  - HTML is generated from Markdown — no extra template engine required.
  - Raises ReportError on unrecoverable problems.
"""

from __future__ import annotations

import html
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.state.workflow_state import WorkflowState, NodeResult
from app.enums.workflow_status import WorkflowNodeStatus
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── CSS for the HTML report ───────────────────────────────────────────────────
_HTML_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 40px auto; padding: 0 24px;
       color: #1a1a2e; background: #f8f9fa; }
h1   { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }
h2   { color: #0f3460; border-bottom: 1px solid #dee2e6; padding-bottom: 4px;
       margin-top: 36px; }
h3   { color: #533483; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.9em; }
th    { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; }
td    { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
tr:nth-child(even) td { background: #e9ecef; }
tr.best td { background: #d4edda; font-weight: 600; }
code  { background: #e9ecef; padding: 2px 6px; border-radius: 3px;
        font-family: 'Courier New', monospace; font-size: 0.88em; }
pre   { background: #1a1a2e; color: #e0e0e0; padding: 16px; border-radius: 6px;
        overflow-x: auto; font-size: 0.85em; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-size: 0.8em; font-weight: 600; }
.badge-best  { background: #28a745; color: #fff; }
.badge-good  { background: #17a2b8; color: #fff; }
.badge-other { background: #6c757d; color: #fff; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px; margin: 16px 0; }
.summary-card { background: #fff; border: 1px solid #dee2e6; border-radius: 8px;
                padding: 16px; text-align: center; }
.summary-card .value { font-size: 2em; font-weight: 700; color: #0f3460; }
.summary-card .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid #dee2e6;
         font-size: 0.8em; color: #6c757d; text-align: center; }
"""


# ── Domain exception ──────────────────────────────────────────────────────────

class ReportError(Exception):
    """Raised when the Report Agent cannot complete its work."""


# ── Agent callable ────────────────────────────────────────────────────────────

def report_agent(state: WorkflowState) -> dict[str, Any]:
    """
    LangGraph node function for the Report Agent.

    Reads from state:
        All prior agent outputs (plan, eda_summary, cleaning_report,
        feature_report, training_metrics, evaluation_results, node_results, …)

    Writes to state:
        report_markdown_path, report_html_path, node_results (appended)
    """
    start = time.perf_counter()
    node_name = "report"
    logger.info(f"[{node_name}] Starting — generating pipeline report")

    try:
        result = _run_report(state)
        duration = round(time.perf_counter() - start, 3)
        node_result: NodeResult = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.DONE,
            "duration_seconds": duration,
            "output_summary": {
                "markdown_path": result["report_markdown_path"],
                "html_path": result["report_html_path"],
            },
        }
        logger.info(
            f"[{node_name}] Done in {duration}s — "
            f"md={result['report_markdown_path']}"
        )

    except (ReportError, OSError) as exc:
        duration = round(time.perf_counter() - start, 3)
        node_result = {
            "node_name": node_name,
            "status": WorkflowNodeStatus.ERROR,
            "duration_seconds": duration,
            "error": str(exc),
        }
        logger.error(f"[{node_name}] Failed: {exc}")
        existing = list(state.get("node_results") or [])
        existing.append(node_result)
        errors = list(state.get("errors") or [])
        errors.append(str(exc))
        return {
            "node_results": existing,
            "errors": errors,
            "report_markdown_path": "",
            "report_html_path": "",
        }

    existing = list(state.get("node_results") or [])
    existing.append(node_result)
    return {**result, "node_results": existing}


# ── Core report logic ─────────────────────────────────────────────────────────

def _run_report(state: WorkflowState) -> dict[str, Any]:
    """
    Pure report generation logic — separated from the node wrapper for testability.
    Returns a dict ready to be merged into WorkflowState.
    """
    experiment_id = state.get("experiment_id") or uuid.uuid4().hex

    # ── 1. Render Markdown ────────────────────────────────────────────────────
    md = _render_markdown(state)

    # ── 2. Convert to HTML ────────────────────────────────────────────────────
    html_content = _markdown_to_html(md, experiment_id, state)

    # ── 3. Save files ─────────────────────────────────────────────────────────
    import app.utils.file_manager as _fm
    _fm.REPORTS_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    _fm.REPORTS_HTML_DIR.mkdir(parents=True, exist_ok=True)

    md_path = _fm.REPORTS_MARKDOWN_DIR / f"{experiment_id}.md"
    html_path = _fm.REPORTS_HTML_DIR / f"{experiment_id}.html"

    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")

    logger.info(f"[report] Saved Markdown → {md_path}")
    logger.info(f"[report] Saved HTML    → {html_path}")

    return {
        "report_markdown_path": str(md_path),
        "report_html_path": str(html_path),
    }


# ── Markdown renderer ─────────────────────────────────────────────────────────

def _render_markdown(state: WorkflowState) -> str:
    """Build the full Markdown report string from WorkflowState."""
    sections: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    exp_id = state.get("experiment_id", "unknown")
    target = state.get("target_column", "unknown")
    task_type = state.get("task_type")
    task_str = task_type.value if task_type else "unknown"

    # ── Header ────────────────────────────────────────────────────────────────
    sections.append(f"# Autonomous Data Science Team — Pipeline Report\n")
    sections.append(
        f"**Experiment ID:** `{exp_id}`  \n"
        f"**Generated:** {now}  \n"
        f"**Target Column:** `{target}`  \n"
        f"**Task Type:** `{task_str}`\n"
    )

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    sections.append("## 1. Executive Summary\n")
    ev = state.get("evaluation_results") or {}
    best_model = state.get("best_model_name") or ev.get("best_model", "N/A")
    best_score = ev.get("best_score", "N/A")
    primary_metric = ev.get("primary_metric", "N/A")
    n_models = len(state.get("training_metrics") or {})
    eda = state.get("eda_summary") or {}
    shape = eda.get("shape", {})
    n_rows = shape.get("rows", "N/A")
    n_cols = shape.get("columns", "N/A")

    sections.append(
        f"| Item | Value |\n"
        f"|---|---|\n"
        f"| Dataset rows | {n_rows} |\n"
        f"| Dataset columns | {n_cols} |\n"
        f"| Models trained | {n_models} |\n"
        f"| Best model | **{best_model}** |\n"
        f"| {primary_metric} (best) | **{best_score}** |\n"
    )

    # ── 2. Dataset Overview ───────────────────────────────────────────────────
    sections.append("## 2. Dataset Overview\n")
    if eda:
        numeric_cols = eda.get("numeric_columns", [])
        cat_cols = eda.get("categorical_columns", [])
        missing_cols = eda.get("missing_columns", [])
        sections.append(
            f"- **Rows:** {n_rows}  \n"
            f"- **Columns:** {n_cols}  \n"
            f"- **Numeric features:** {len(numeric_cols)} — "
            f"`{'`, `'.join(numeric_cols[:8])}{'…' if len(numeric_cols) > 8 else ''}`  \n"
            f"- **Categorical features:** {len(cat_cols)} — "
            f"`{'`, `'.join(cat_cols[:8])}{'…' if len(cat_cols) > 8 else ''}`  \n"
            f"- **Columns with missing values:** {len(missing_cols)}"
            + (f" — `{'`, `'.join(missing_cols)}`" if missing_cols else "") + "\n"
        )
        # Target analysis
        ta = eda.get("target_analysis", {})
        if ta:
            sections.append(f"\n### Target Column: `{target}`\n")
            if ta.get("type") == "categorical":
                cc = ta.get("class_counts", {})
                cb = ta.get("class_balance", {})
                rows = "\n".join(
                    f"| {cls} | {cc.get(cls, 0)} | {cb.get(cls, 0):.1f}% |"
                    for cls in cc
                )
                sections.append(
                    f"| Class | Count | Balance |\n|---|---|---|\n{rows}\n"
                )
            else:
                sections.append(
                    f"| Stat | Value |\n|---|---|\n"
                    f"| Mean | {ta.get('mean', 'N/A')} |\n"
                    f"| Std | {ta.get('std', 'N/A')} |\n"
                    f"| Min | {ta.get('min', 'N/A')} |\n"
                    f"| Max | {ta.get('max', 'N/A')} |\n"
                    f"| Median | {ta.get('median', 'N/A')} |\n"
                )
    else:
        sections.append("_EDA summary not available._\n")

    # ── 3. Data Quality ───────────────────────────────────────────────────────
    sections.append("## 3. Data Quality\n")
    missing_report = eda.get("missing_report", {})
    if missing_report:
        rows = "\n".join(
            f"| `{col}` | {info['count']} | {info['pct']:.1f}% |"
            for col, info in missing_report.items()
        )
        sections.append(
            f"| Column | Missing Count | Missing % |\n|---|---|---|\n{rows}\n"
        )
    else:
        sections.append("✅ No missing values detected in the dataset.\n")

    # ── 4. Cleaning Summary ───────────────────────────────────────────────────
    sections.append("## 4. Cleaning Summary\n")
    cr = state.get("cleaning_report") or {}
    if cr:
        sections.append(
            f"| Metric | Value |\n|---|---|\n"
            f"| Rows before | {cr.get('rows_before', 'N/A')} |\n"
            f"| Rows after | {cr.get('rows_after', 'N/A')} |\n"
            f"| Rows dropped | {cr.get('rows_dropped', 0)} |\n"
            f"| Duplicates removed | {cr.get('duplicates_removed', 0)} |\n"
            f"| Target rows dropped | {cr.get('target_missing_dropped', 0)} |\n"
            f"| Columns dropped | {len(cr.get('columns_dropped', []))} |\n"
            f"| Numeric columns imputed | {len(cr.get('columns_imputed_numeric', []))} |\n"
            f"| Categorical columns imputed | {len(cr.get('columns_imputed_categorical', []))} |\n"
            f"| Columns clipped (outliers) | {len(cr.get('columns_clipped', []))} |\n"
        )
        if cr.get("columns_dropped"):
            sections.append(
                f"\n**Dropped columns:** "
                f"`{'`, `'.join(cr['columns_dropped'])}`\n"
            )
    else:
        sections.append("_Cleaning report not available._\n")

    # ── 5. Feature Engineering Summary ───────────────────────────────────────
    sections.append("## 5. Feature Engineering Summary\n")
    fr = state.get("feature_report") or {}
    if fr:
        sections.append(
            f"| Metric | Value |\n|---|---|\n"
            f"| Features before | {fr.get('features_before', 'N/A')} |\n"
            f"| Features after | {fr.get('features_after', 'N/A')} |\n"
            f"| Numeric columns scaled | {len(fr.get('numeric_scaled', []))} |\n"
            f"| Ordinal-encoded columns | {len(fr.get('categorical_ordinal_encoded', []))} |\n"
            f"| One-hot encoded columns | {len(fr.get('categorical_ohe_encoded', []))} |\n"
            f"| Low-variance features dropped | {len(fr.get('features_dropped_low_variance', []))} |\n"
            f"| Top-k selection (k) | {fr.get('top_k', 'N/A')} |\n"
        )
        sel = fr.get("features_selected", [])
        if sel:
            sections.append(
                f"\n**Selected features ({len(sel)}):** "
                f"`{'`, `'.join(str(f) for f in sel[:15])}"
                f"{'…' if len(sel) > 15 else ''}`\n"
            )
    else:
        sections.append("_Feature engineering report not available._\n")

    # ── 6. Model Training Results ─────────────────────────────────────────────
    sections.append("## 6. Model Training Results\n")
    training_metrics: dict[str, Any] = state.get("training_metrics") or {}
    if training_metrics:
        # Collect all metric keys (excluding bookkeeping)
        _BK = {"train_duration_seconds", "n_train", "n_test"}
        all_keys: list[str] = []
        for m in training_metrics.values():
            for k in m:
                if k not in _BK and k not in all_keys:
                    all_keys.append(k)

        header = "| Model | " + " | ".join(all_keys) + " | Duration (s) |"
        sep = "|---|" + "---|" * len(all_keys) + "---|"
        rows_list = [header, sep]
        for model_name, metrics in training_metrics.items():
            vals = " | ".join(
                str(metrics.get(k, "N/A")) for k in all_keys
            )
            dur = metrics.get("train_duration_seconds", "N/A")
            rows_list.append(f"| {model_name} | {vals} | {dur} |")
        sections.append("\n".join(rows_list) + "\n")
    else:
        sections.append("_Training metrics not available._\n")

    # ── 7. Model Evaluation & Ranking ─────────────────────────────────────────
    sections.append("## 7. Model Evaluation & Ranking\n")
    if ev:
        pm = ev.get("primary_metric", "score")
        hib = ev.get("higher_is_better", True)
        direction = "↑ higher is better" if hib else "↓ lower is better"
        sections.append(f"**Primary metric:** `{pm}` ({direction})\n")

        ranked = ev.get("ranked_models", [])
        if ranked:
            header = f"| Rank | Model | {pm} |"
            sep = "|---|---|---|"
            rows_list = [header, sep]
            for entry in ranked:
                medal = "🥇" if entry["rank"] == 1 else (
                    "🥈" if entry["rank"] == 2 else "🥉" if entry["rank"] == 3 else ""
                )
                rows_list.append(
                    f"| {entry['rank']} {medal} | {entry['model']} "
                    f"| **{entry['primary_score']}** |"
                )
            sections.append("\n".join(rows_list) + "\n")
    else:
        sections.append("_Evaluation results not available._\n")

    # ── 8. Best Model ─────────────────────────────────────────────────────────
    sections.append("## 8. Best Model\n")
    if best_model and best_model != "N/A":
        best_path = state.get("best_model_path", "")
        sections.append(
            f"**Model:** `{best_model}`  \n"
            f"**{primary_metric}:** `{best_score}`  \n"
            f"**Artifact:** `{best_path}`\n"
        )
        # Full metrics for best model
        bm_metrics = training_metrics.get(best_model, {})
        if bm_metrics:
            _BK = {"train_duration_seconds", "n_train", "n_test"}
            rows_list = ["| Metric | Value |", "|---|---|"]
            for k, v in bm_metrics.items():
                if k not in _BK:
                    rows_list.append(f"| {k} | {v} |")
            sections.append("\n".join(rows_list) + "\n")
    else:
        sections.append("_Best model information not available._\n")

    # ── 9. Pipeline Execution Timeline ────────────────────────────────────────
    sections.append("## 9. Pipeline Execution Timeline\n")
    node_results: list[dict] = state.get("node_results") or []
    if node_results:
        header = "| Step | Node | Status | Duration (s) |"
        sep = "|---|---|---|---|"
        rows_list = [header, sep]
        for i, nr in enumerate(node_results, 1):
            status_icon = {
                "done": "✅", "error": "❌", "skipped": "⏭️",
                "in_progress": "🔄", "waiting": "⏳",
            }.get(str(nr.get("status", "")).lower(), "❓")
            dur = nr.get("duration_seconds", "N/A")
            rows_list.append(
                f"| {i} | {nr.get('node_name', 'unknown')} "
                f"| {status_icon} {nr.get('status', 'unknown')} "
                f"| {dur} |"
            )
        sections.append("\n".join(rows_list) + "\n")
    else:
        sections.append("_No execution timeline available._\n")

    # ── 10. Appendix ──────────────────────────────────────────────────────────
    sections.append("## 10. Appendix\n")

    chart_paths: list[str] = state.get("chart_paths") or []
    if chart_paths:
        sections.append(f"### Charts ({len(chart_paths)} generated)\n")
        for cp in chart_paths[:20]:
            sections.append(f"- `{cp}`")
        if len(chart_paths) > 20:
            sections.append(f"- _…and {len(chart_paths) - 20} more_")
        sections.append("")

    trained_paths: dict[str, str] = state.get("trained_model_paths") or {}
    if trained_paths:
        sections.append("### Model Artifacts\n")
        for name, path in trained_paths.items():
            sections.append(f"- **{name}:** `{path}`")
        sections.append("")

    sections.append(
        f"\n---\n_Report generated by Autonomous Data Science Team "
        f"on {now}_\n"
    )

    return "\n".join(sections)


# ── HTML converter ────────────────────────────────────────────────────────────

def _markdown_to_html(md: str, experiment_id: str, state: WorkflowState) -> str:
    """
    Convert Markdown to a self-contained HTML document.
    Uses a simple line-by-line parser — no external dependencies required.
    """
    best_model = state.get("best_model_name", "")
    lines = md.split("\n")
    body_parts: list[str] = []
    in_table = False
    in_code = False
    table_header_done = False
    row_count = 0

    for line in lines:
        # Code blocks
        if line.startswith("```"):
            if in_code:
                body_parts.append("</code></pre>")
                in_code = False
            else:
                body_parts.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            body_parts.append(html.escape(line))
            continue

        # Headings
        if line.startswith("#### "):
            body_parts.append(f"<h4>{_inline_md(line[5:])}</h4>")
        elif line.startswith("### "):
            body_parts.append(f"<h3>{_inline_md(line[4:])}</h3>")
        elif line.startswith("## "):
            body_parts.append(f"<h2>{_inline_md(line[3:])}</h2>")
        elif line.startswith("# "):
            body_parts.append(f"<h1>{_inline_md(line[2:])}</h1>")

        # Tables
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c.replace("-", "").replace(":", "").strip()) == set() or
                   c.strip() in ("", "---", ":---", "---:") for c in cells):
                # Separator row — close thead, open tbody
                if not table_header_done:
                    body_parts.append("</tr></thead><tbody>")
                    table_header_done = True
                    row_count = 0
            else:
                if not in_table:
                    body_parts.append('<table>')
                    body_parts.append("<thead><tr>")
                    in_table = True
                    table_header_done = False
                    tag = "th"
                elif not table_header_done:
                    tag = "th"
                else:
                    # Highlight best model row
                    is_best = any(
                        best_model and best_model in c for c in cells
                    )
                    row_class = ' class="best"' if is_best else ""
                    body_parts.append(f"<tr{row_class}>")
                    tag = "td"
                    row_count += 1

                for cell in cells:
                    body_parts.append(f"<{tag}>{_inline_md(cell)}</{tag}>")

                if tag == "th" and not table_header_done:
                    pass  # header row — don't close yet
                elif tag == "td":
                    body_parts.append("</tr>")

        else:
            # Close table if we were in one
            if in_table:
                body_parts.append("</tbody></table>")
                in_table = False
                table_header_done = False

            # Horizontal rule
            if line.strip() in ("---", "***", "___"):
                body_parts.append("<hr>")
            # List items
            elif line.startswith("- ") or line.startswith("* "):
                body_parts.append(f"<li>{_inline_md(line[2:])}</li>")
            # Blank line
            elif line.strip() == "":
                body_parts.append("")
            # Paragraph
            else:
                body_parts.append(f"<p>{_inline_md(line)}</p>")

    if in_table:
        body_parts.append("</tbody></table>")

    body = "\n".join(body_parts)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pipeline Report — {html.escape(experiment_id)}</title>
  <style>{_HTML_CSS}</style>
</head>
<body>
{body}
<footer>Autonomous Data Science Team &bull; {html.escape(now)}</footer>
</body>
</html>"""


def _inline_md(text: str) -> str:
    """Convert inline Markdown (bold, italic, code, links) to HTML."""
    import re
    # Escape HTML first
    text = html.escape(text)
    # Bold **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic *text*
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code `text`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Line breaks (two trailing spaces)
    text = text.replace("  ", "<br>")
    return text
