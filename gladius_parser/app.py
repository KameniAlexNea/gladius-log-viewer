"""Gladius Log Viewer — tree-aware Gradio UI.

The view renders the execution as a two-level tree:

  ┌─ gladius (root)  ─────────────────────────────────────────────────────────┐
  │  [preamble events: startup reads, thinking, messages]                     │
  │  ├─ 🤖 Task → team-lead  [plan]  ──── 42s · 3 tools · 0 err    [▶ expand]│
  │  │  thinking / tool calls / results                                        │
  │  ├─ 🤖 Task → data-expert  [EDA]  ─── 2m 3s · 18 tools · 1 err           │
  │  │  ...                                                                    │
  │  [root events between tasks: thinking, messages, todos]                   │
  │  └─ ✅  done  status=OK  turns=61  cost=$13.04                             │
  └───────────────────────────────────────────────────────────────────────────┘

Usage:
    python app.py
    python app.py --log /path/to/gladius.log
"""

from __future__ import annotations

import dataclasses
import html
import json
import re
from pathlib import Path

import gradio as gr

from .parser import (
    AGENT_START, MESSAGE, OTHER,
    RESULT_CONT, RESULT_ERR, RESULT_OK, SESSION,
    STATUS, TASK, THINKING, TODO_ITEM, TODO_WRITE, TOOL_USE, WARNING,
    AgentNode, Event, RootNode, parse_log,
)
from .css import CSS as _CSS, AGENT_COLOURS as _AGENT_COLOURS

# ═══════════════════════════════════════════ HELPERS ══════════════════════════

def _e(s: str) -> str:
    return html.escape(str(s))

def _ts(ev: Event) -> str:
    return ev.ts.strftime("%H:%M:%S")

def _trunc(s: str, n: int = 120) -> str:
    return s[:n] + "…" if len(s) > n else s

def _fmt_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s), indent=2)
    except Exception:
        return s

def _dur(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _agent_colour(name: str, seen: dict[str, str]) -> str:
    if name not in seen:
        seen[name] = _AGENT_COLOURS[len(seen) % len(_AGENT_COLOURS)]
    return seen[name]

# ═══════════════════════════════════════════ EXTRACTORS ═══════════════════════

def _ex_tools(text: str) -> list[str]:
    m = re.search(r"tools=\[(.*)\]", text)
    if not m:
        return []
    return [t.strip().strip("'\"") for t in m.group(1).split(",")]

def _ex_session(text: str) -> str:
    m = re.search(r"session=(.*)", text)
    return m.group(1).strip() if m else text

def _ex_thinking(text: str) -> str:
    m = re.search(r"\(thinking\)(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text

def _ex_message(text: str) -> str:
    m = re.search(r"💬 \[gladius\] (.*)", text)
    return m.group(1) if m else text

def _ex_tool(text: str, is_sub: bool) -> tuple[str, str]:
    # Subagent tools: "🔧 [gladius] ➣<name> Bash  {args}" → strip up to agent badge
    t = re.sub(r".*➣\w+\s*", "", text) if is_sub else re.sub(r"🔧\s*\[gladius\]\s*", "", text)
    if "  " in t:
        idx = t.index("  ")
        return t[:idx].strip(), t[idx + 2:].strip()
    return t.strip(), ""

def _ex_result_text(text: str, kind: str) -> str:
    idx = text.find("[gladius]")
    if idx >= 0:
        return text[idx + len("[gladius]"):].strip()
    marker = "✓" if kind == RESULT_OK else "✗"
    idx = text.find(marker)
    return text[idx + 1:].strip() if idx >= 0 else text

def _ex_status(text: str) -> str:
    return re.sub(r"━+\s*\[gladius\]\s*", "", text).strip()

def _ex_todo_item(text: str) -> tuple[str, str]:
    s = text.strip()
    for ch in ("✅", "⬜", "🔧"):
        if s.startswith(ch):
            return ch, s[len(ch):].strip()
    return "?", s

# ═══════════════════════════════════════════ RESULT GROUPING ══════════════════

def _group_results(events: list[Event]) -> list[Event | list[Event]]:
    """Merge each RESULT_OK/ERR with its following RESULT_CONT lines."""
    out: list[Event | list[Event]] = []
    current: list[Event] | None = None
    for ev in events:
        if ev.kind in (RESULT_OK, RESULT_ERR):
            if current is not None:
                out.append(current)
            current = [ev]
        elif ev.kind == RESULT_CONT:
            if current is not None:
                current.append(ev)
        else:
            if current is not None:
                out.append(current)
                current = None
            out.append(ev)
    if current is not None:
        out.append(current)
    return out


_MERGE_KINDS = {MESSAGE, THINKING}

def _extract_text(ev: Event) -> str:
    """Extract the human-readable payload from a MESSAGE or THINKING event."""
    if ev.kind == MESSAGE:
        m = re.search(r"💬 \[gladius\][\S]* (.*)", ev.text, re.DOTALL)
        return m.group(1).strip() if m else ev.text
    if ev.kind == THINKING:
        m = re.search(r"\(thinking\)(.*)", ev.text, re.DOTALL)
        return m.group(1).strip() if m else ev.text
    return ev.text


def _merge_consecutive(
    items: list[Event | list[Event]],
) -> list[Event | list[Event]]:
    """Collapse runs of consecutive MESSAGE or THINKING single-events into one.

    Adjacent lines that are the same kind and same is_subagent flag are joined
    so they render as a single row instead of many fragmented ones.
    """
    out: list[Event | list[Event]] = []
    for item in items:
        if (
            isinstance(item, Event)
            and item.kind in _MERGE_KINDS
            and out
            and isinstance(out[-1], Event)
            and out[-1].kind == item.kind
            and out[-1].is_subagent == item.is_subagent
        ):
            prev = out[-1]
            joined = _extract_text(prev) + " " + _extract_text(item)
            # Rebuild a synthetic text that the existing renderers can parse.
            if item.kind == MESSAGE:
                badge = " ➣subagent" if item.is_subagent else ""
                new_text = f"💬 [gladius]{badge} {joined}"
            else:  # THINKING
                badge = " ➣subagent" if item.is_subagent else ""
                new_text = f"🧠 [gladius]{badge} (thinking) {joined}"
            out[-1] = dataclasses.replace(prev, text=new_text)
        else:
            out.append(item)
    return out

# ═══════════════════════════════════════════ ROW RENDERERS ════════════════════

def _row(ts: str, icon: str, css: str, body: str) -> str:
    return (
        f'<div class="lv-ev {css}">'
        f'<div class="lv-ts">{ts}</div>'
        f'<div class="lv-icon">{icon}</div>'
        f'<div class="lv-body">{body}</div>'
        f"</div>"
    )

def _collapsible(summary_html: str, full_text: str) -> str:
    return (
        f'<details class="lv-col">'
        f"<summary>{summary_html}</summary>"
        f'<pre class="lv-pre">{_e(full_text)}</pre>'
        f"</details>"
    )

# ── per-kind row builders ─────────────────────────────────────────────────────

def _r_agent_start(ev: Event) -> str:
    tools = _ex_tools(ev.text)
    badges = "".join(f'<span class="tbadge">{_e(t)}</span>' for t in tools)
    return _row(_ts(ev), "▶", "ev-start",
                f'<span style="font-weight:700;color:#4f46e5">agent started</span>'
                f'<div class="tbadges">{badges}</div>')

def _r_session(ev: Event) -> str:
    return _row(_ts(ev), "🔑", "ev-session",
                f'<span class="dim">session {_e(_ex_session(ev.text))}</span>')

def _r_thinking(ev: Event) -> str:
    sub  = '<span class="sub-badge">sub</span>' if ev.is_subagent else ""
    txt  = _ex_thinking(ev.text)
    if len(txt) > 120:
        body = sub + _collapsible(f'<span class="preview think-txt">{_e(_trunc(txt))}</span>', txt)
    else:
        body = f'{sub}<span class="think-txt">{_e(txt)}</span>'
    return _row(_ts(ev), "🧠", "ev-think", body)

def _r_message(ev: Event) -> str:
    sub = '<span class="sub-badge">sub</span>' if ev.is_subagent else ""
    return _row(_ts(ev), "💬", "ev-msg",
                f'{sub}<span class="msg-txt">{_e(_ex_message(ev.text))}</span>')

def _r_tool_use(ev: Event) -> str:
    name, args = _ex_tool(ev.text, ev.is_subagent)
    sub = '<span class="sub-badge">sub</span>' if ev.is_subagent else ""
    css = "ev-tool-s" if ev.is_subagent else "ev-tool"
    nm  = f'<span class="tool-name">{_e(name)}</span>'
    if args:
        body = sub + nm + _collapsible(
            f'<span class="preview dim">{_e(_trunc(args, 100))}</span>',
            _fmt_json(args),
        )
    else:
        body = sub + nm
    return _row(_ts(ev), "🔧", css, body)

def _r_todo_write(ev: Event) -> str:
    m = re.search(r"TodoWrite\s+(.*)", ev.text)
    prog = m.group(1).strip() if m else ""
    return _row(_ts(ev), "📋", "ev-todo",
                f'<span style="font-weight:600;color:#7c3aed">Todo: {_e(prog)}</span>')

def _r_todo_item(ev: Event) -> str:
    status, text = _ex_todo_item(ev.text)
    if status == "✅":
        icon, css = "✅", "item-done"
    elif status == "🔧":
        icon, css = "🔄", "item-curr"
    else:
        icon, css = "⬜", "item-pend"
    return _row(_ts(ev), icon, "ev-item",
                f'<span class="{css}">{_e(text)}</span>')

def _r_result_group(group: list[Event]) -> str:
    if not group:
        return ""
    head = group[0]
    is_ok = head.kind == RESULT_OK
    icon = "✓" if is_ok else "✗"
    css  = "ev-ok" if is_ok else "ev-err"
    tc   = "ok-txt" if is_ok else "err-txt"

    first = _ex_result_text(head.text, head.kind)
    conts = [e.text.strip() for e in group[1:] if e.kind == RESULT_CONT]
    all_lines = [first] + conts
    full = "\n".join(all_lines)

    if conts or len(first) > 120:
        n_label = f" · {len(all_lines)} lines" if len(all_lines) > 1 else ""
        body = _collapsible(
            f'<span class="preview {tc}">{_e(_trunc(first))}</span>'
            f'<span class="dim">{_e(n_label)}</span>',
            full,
        )
    else:
        body = f'<span class="{tc}">{_e(first)}</span>'
    return _row(_ts(head), icon, css, body)

def _r_status(ev: Event) -> str:
    return _row(_ts(ev), "✅", "ev-status",
                f'<span style="font-weight:700;color:#0f766e">{_e(_ex_status(ev.text))}</span>')

def _r_warning(ev: Event) -> str:
    txt = re.sub(r"^\[gladius\]\s*", "", ev.text).strip()
    return _row(_ts(ev), "⚠", "ev-warn",
                f'<span style="color:#92400e">{_e(txt)}</span>')

# ── dispatch ──────────────────────────────────────────────────────────────────

_KIND_DISPATCH = {
    AGENT_START: _r_agent_start,
    SESSION:     _r_session,
    THINKING:    _r_thinking,
    MESSAGE:     _r_message,
    TOOL_USE:    _r_tool_use,
    TODO_WRITE:  _r_todo_write,
    TODO_ITEM:   _r_todo_item,
    STATUS:      _r_status,
    WARNING:     _r_warning,
    RESULT_OK:   lambda e: _r_result_group([e]),
    RESULT_ERR:  lambda e: _r_result_group([e]),
    RESULT_CONT: lambda _: "",
    OTHER:       lambda _: "",
    TASK:         lambda _: "",
}

def _render_event(item: Event | list[Event]) -> str:
    if isinstance(item, list):
        return _r_result_group(item)
    fn = _KIND_DISPATCH.get(item.kind, lambda _: "")
    return fn(item)

def _render_events(events: list[Event]) -> str:
    grouped = _merge_consecutive(_group_results(events))
    return "".join(_render_event(item) for item in grouped)

# ═══════════════════════════════════════════ AGENT NODE BLOCK ═════════════════

def _render_agent_node(node: AgentNode, colour: str, index: int) -> str:
    if not node.events:
        title = f" · {_e(node.title)}" if node.title else ""
        return (
            f'<div class="lv-agent-retry">'
            f'<span class="retry-name">{_e(node.agent_name)}</span>'
            f'<span>{title}</span>'
            f'<span style="margin-left:8px;font-style:italic">↺ re-dispatched</span>'
            f"</div>"
        )

    dur   = _dur(node.duration_s)
    tools = node.n_tools
    errs  = node.n_errors

    err_html = f'<span class="err">{errs} err</span>' if errs else ""
    meta = (
        f'<span>{_e(dur)}</span>'
        f'<span>{tools} tools</span>'
        + err_html
    )

    header = (
        f'<div class="lv-agent-header" onclick="'
        f'var d=this.nextElementSibling; d.open=!d.open; '
        f'this.querySelector(\'.lv-toggle\').textContent=d.open?\'▾\':\'▸\'">'
        f'<span class="lv-agent-name">{_e(node.agent_name)}</span>'
        f'<span class="lv-agent-title">{_e(node.title)}</span>'
        f'<div class="lv-agent-meta">{meta}</div>'
        f'<span class="lv-toggle">▸</span>'
        f"</div>"
    )

    inner_html = _render_events(node.events)
    details = (
        f'<details class="lv-agent-details">'
        f'<summary></summary>'
        f'<div class="lv-agent-events">{inner_html}</div>'
        f"</details>"
    )

    return (
        f'<div class="lv-agent-block" style="--agent-color:{colour}">'
        f"{header}{details}"
        f"</div>"
    )

# ═══════════════════════════════════════════ ROOT NODE RENDER ════════════════

# Inline onclick for tab buttons — self-contained, no global function needed.
_ITER_BTN_ONCLICK = (
    "var w=this.closest('.lv-tab-wrapper');"
    "w.querySelectorAll('.lv-tab-btn').forEach(function(b){b.classList.remove('lv-tab-active')});"
    "w.querySelectorAll('.lv-tab-pane').forEach(function(p){p.style.display='none'});"
    "this.classList.add('lv-tab-active');"
    "w.querySelector('#'+this.getAttribute('data-pane')).style.display='';"
)


def _split_iterations(
    children: list,
) -> list[list]:
    """Group root.children into per-iteration slices, each ending at a STATUS event."""
    groups: list[list] = []
    current: list = []
    for child in children:
        current.append(child)
        if isinstance(child, Event) and child.kind == STATUS:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _iter_label(idx: int, children: list) -> str:
    """Short tab label for one iteration."""
    label = f"Run {idx + 1}"
    for child in reversed(children):
        if isinstance(child, Event) and child.kind == STATUS:
            m = re.search(r"status=(\S+)", _ex_status(child.text))
            if m:
                label += f" · {m.group(1)}"
            break
    return label



def _render_children_section(
    children: list,
    colour_map: dict[str, str],
) -> str:
    """Render a flat list of AgentNode/Event children into HTML."""
    parts: list[str] = []
    ev_run: list[Event] = []

    def _flush() -> None:
        if ev_run:
            parts.append(_render_events(list(ev_run)))
            ev_run.clear()

    for child in children:
        if isinstance(child, AgentNode):
            _flush()
            colour = _agent_colour(child.agent_name, colour_map)
            idx    = len(colour_map) - 1
            parts.append(_render_agent_node(child, colour, idx))
        else:
            ev_run.append(child)
    _flush()
    return "".join(parts)


def _render_root(root: RootNode) -> str:
    colour_map: dict[str, str] = {}

    stat_parts: list[str] = []
    def stat(n, label: str, colour: str = "") -> None:
        style = f' style="color:{colour}"' if colour else ""
        stat_parts.append(
            f'<div class="lv-stat">'
            f'<span class="lv-stat-n"{style}>{_e(str(n))}</span>'
            f'<span class="lv-stat-lbl">{_e(label)}</span>'
            f"</div>"
        )

    stat(_dur(root.duration_s), "duration")
    stat(root.n_tasks, "tasks")
    stat(root.n_tools_total, "tool calls")
    if root.n_errors_total:
        stat(root.n_errors_total, "errors", "#f87171")

    header_html = (
        f'<div class="lv-header">'
        f'<span class="lv-header-title">🚀 {_e(root.agent_name)}</span>'
        + "".join(stat_parts)
        + "</div>"
    )

    body_parts: list[str] = []

    # Preamble events belong to the first iteration tab.
    preamble_html = _render_events(root.preamble) if root.preamble else ""

    # Split children into per-iteration groups
    iterations = _split_iterations(root.children)

    if len(iterations) <= 1:
        # Single iteration — render inline as before
        body_parts.append(
            preamble_html
            + _render_children_section(iterations[0] if iterations else [], colour_map)
        )
    else:
        # Multiple iterations — render as tabs.
        # Iteration 0: prepend preamble, keep AGENT_START/SESSION.
        # Iterations 1+: skip AGENT_START/SESSION restart boilerplate.
        tab_bar_parts = ['<div class="lv-tab-bar">']
        pane_parts: list[str] = []
        for i, iter_children in enumerate(iterations):
            label   = _iter_label(i, iter_children)
            pane_id = f"lv-iter-{i}"
            active  = " lv-tab-active" if i == 0 else ""
            display = "" if i == 0 else "display:none"
            tab_bar_parts.append(
                f'<button class="lv-tab-btn{active}" data-pane="{pane_id}"'
                f' onclick="{_ITER_BTN_ONCLICK}">{_e(label)}</button>'
            )
            prefix = preamble_html if i == 0 else ""
            pane_html = prefix + _render_children_section(
                iter_children, colour_map
            )
            pane_parts.append(
                f'<div class="lv-tab-pane" id="{pane_id}" style="{display}">'
                f'{pane_html}</div>'
            )
        tab_bar_parts.append("</div>")
        body_parts.append(
            '<div class="lv-tab-wrapper">'
            + "".join(tab_bar_parts)
            + "".join(pane_parts)
            + "</div>"
        )

    # Epilogue
    if root.epilogue:
        body_parts.append('<div class="lv-section-lbl">epilogue</div>')
        body_parts.append(_render_events(root.epilogue))

    body = '<div class="lv-root">' + "".join(body_parts) + "</div>"
    return _CSS + '<div class="lv">' + header_html + body + "</div>"

# ═══════════════════════════════════════════ PUBLIC RENDER ════════════════════

def render_log(content: str) -> str:
    root = parse_log(content)
    return _render_root(root)

# ═══════════════════════════════════════════ GRADIO APP ═══════════════════════

_PLACEHOLDER = (
    '<div style="padding:60px;text-align:center;color:#94a3b8;font-family:sans-serif">'
    "Enter the path to a <code>gladius.log</code> file and click <b>Load</b>, press Enter, or upload a file."
    "</div>"
)

# Maximum number of iteration tabs to pre-create in the UI.
MAX_ITER = 12


def _render_header_html(root: RootNode) -> str:
    """Stats header bar as standalone HTML. Also injects CSS once for the whole page."""
    stat_parts: list[str] = []

    def stat(n, label: str, colour: str = "") -> None:
        style = f' style="color:{colour}"' if colour else ""
        stat_parts.append(
            f'<div class="lv-stat">'
            f'<span class="lv-stat-n"{style}>{_e(str(n))}</span>'
            f'<span class="lv-stat-lbl">{_e(label)}</span>'
            f"</div>"
        )

    stat(_dur(root.duration_s), "duration")
    stat(root.n_tasks, "tasks")
    stat(root.n_tools_total, "tool calls")
    if root.n_errors_total:
        stat(root.n_errors_total, "errors", "#f87171")
    return (
        _CSS
        + f'<div class="lv-header">'
        f'<span class="lv-header-title">🚀 {_e(root.agent_name)}</span>'
        + "".join(stat_parts)
        + "</div>"
    )


def build_ui(default_path: str = "") -> gr.Blocks:
    with gr.Blocks(title="Gladius Log Viewer") as demo:
        gr.Markdown("## 🔍 Gladius Log Viewer")
        with gr.Row():
            path_box = gr.Textbox(
                value=default_path, label="Log file path",
                placeholder="/path/to/gladius.log", scale=6,
            )
            load_btn = gr.Button("Load", variant="primary", scale=1)
        with gr.Row():
            upload_box = gr.File(
                label="Or upload a log file",
                file_types=[".log", ".txt"],
                type="filepath",
            )

        # Stats header (with CSS injection) — shown after a log is loaded.
        header_out = gr.HTML(value=_PLACEHOLDER)
        # Preamble (startup events), shown only when present.
        preamble_out = gr.HTML(value="", visible=False)

        # Pre-create MAX_ITER native Gradio tabs; unused ones are hidden.
        tab_objs: list[gr.Tab] = []
        html_objs: list[gr.HTML] = []
        with gr.Column(visible=False) as tabs_col:
            with gr.Tabs():
                for i in range(MAX_ITER):
                    with gr.Tab(f"Run {i + 1}", visible=False) as t:
                        h = gr.HTML(value="")
                    tab_objs.append(t)
                    html_objs.append(h)

        # Epilogue (post-run events), shown only when present.
        epilogue_out = gr.HTML(value="", visible=False)

        # ── inner helpers ────────────────────────────────────────────────────

        def _process(content: str) -> list:
            root = parse_log(content)
            colour_map: dict[str, str] = {}

            # Preamble folds into tab 0 — no separate section above the tabs.
            preamble_html = _render_events(root.preamble) if root.preamble else ""

            epilogue_html = ""
            if root.epilogue:
                epilogue_html = (
                    '<div class="lv-section-lbl">epilogue</div>'
                    + _render_events(root.epilogue)
                )

            iterations = _split_iterations(root.children)
            tab_updates: list = []
            html_updates: list = []
            for i in range(MAX_ITER):
                if i < len(iterations):
                    label = _iter_label(i, iterations[i])
                    prefix = preamble_html if i == 0 else ""
                    html = (
                        '<div class="lv">'
                        + prefix
                        + _render_children_section(
                            iterations[i], colour_map
                        )
                        + "</div>"
                    )
                    tab_updates.append(gr.update(visible=True, label=label))
                    html_updates.append(gr.update(value=html))
                else:
                    tab_updates.append(gr.update(visible=False, label=f"Run {i + 1}"))
                    html_updates.append(gr.update(value=""))

            return (
                [_render_header_html(root)]
                + [gr.update(value="", visible=False)]  # preamble_out always hidden
                + [gr.update(visible=bool(iterations))]
                + tab_updates
                + html_updates
                + [gr.update(value=epilogue_html, visible=bool(epilogue_html))]
            )

        def _empty_updates() -> list:
            return (
                [_PLACEHOLDER]
                + [gr.update(value="", visible=False)]
                + [gr.update(visible=False)]
                + [gr.update(visible=False, label=f"Run {i + 1}") for i in range(MAX_ITER)]
                + [gr.update(value="") for _ in range(MAX_ITER)]
                + [gr.update(value="", visible=False)]
            )

        def _err_updates(msg: str) -> list:
            return (
                [msg]
                + [gr.update(value="", visible=False)]
                + [gr.update(visible=False)]
                + [gr.update(visible=False, label=f"Run {i + 1}") for i in range(MAX_ITER)]
                + [gr.update(value="") for _ in range(MAX_ITER)]
                + [gr.update(value="", visible=False)]
            )

        def _handle_path(path: str) -> list:
            path = path.strip()
            if not path:
                return _empty_updates()
            p = Path(path)
            if not p.exists():
                return _err_updates(
                    f'<p style="color:#dc2626;padding:20px">File not found: {_e(path)}</p>'
                )
            try:
                content = p.read_text(errors="replace")
            except Exception as exc:
                return _err_updates(
                    f'<p style="color:#dc2626;padding:20px">Error reading file: {_e(str(exc))}</p>'
                )
            return _process(content)

        def _handle_upload(file_path: str | None) -> list:
            if not file_path:
                return _empty_updates()
            try:
                content = Path(file_path).read_text(errors="replace")
            except Exception as exc:
                return _err_updates(
                    f'<p style="color:#dc2626;padding:20px">Error reading file: {_e(str(exc))}</p>'
                )
            return _process(content)

        outputs = [header_out, preamble_out, tabs_col] + tab_objs + html_objs + [epilogue_out]
        load_btn.click(fn=_handle_path, inputs=path_box, outputs=outputs)
        path_box.submit(fn=_handle_path, inputs=path_box, outputs=outputs)
        upload_box.change(fn=_handle_upload, inputs=upload_box, outputs=outputs)
        if default_path:
            demo.load(fn=lambda: _handle_path(default_path), outputs=outputs)
    return demo

