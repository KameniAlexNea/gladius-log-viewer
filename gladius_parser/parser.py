"""Gladius log parser — tree model.

The log represents a two-level agent hierarchy:

  gladius (root orchestrator)
  └── does its own tool calls / thinking / messages
  └── Task → <agent-name>  ← opens a child node
       └── ➣subagent tool calls / thinking / messages
  └── resumes: more root-level events
  └── Task → <next-agent>
       └── ...

Key encoding in the raw log
---------------------------
  lineno 158 + 🤖       → Task dispatch  → opens a new AgentNode
  lineno 165 + ➣subagent → tool call inside the current subagent
  lineno 165 + no marker → root-level tool call
  lineno 143             → TodoItem (belongs to whoever is active)
  lineno 177             → tool result (belongs to whoever is active)
  lineno 122 + ━━ done  → STATUS (run complete)
  level=WARNING          → SDK warning
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Union

# ── event kinds ───────────────────────────────────────────────────────────────

AGENT_LAUNCH = "agent_launch"
AGENT_START  = "agent_start"
SESSION      = "session"
TASK         = "task"
THINKING     = "thinking"
MESSAGE      = "message"
TOOL_USE     = "tool_use"
TODO_WRITE   = "todo_write"
TODO_ITEM    = "todo_item"
RESULT_OK    = "result_ok"
RESULT_ERR   = "result_err"
RESULT_CONT  = "result_cont"
STATUS       = "status"
WARNING      = "warning"
OTHER        = "other"

_ANSI_RE     = re.compile(r"\x1b\[[0-9;]*m")
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| "
    r"([\w.]+):([\w_]+):(\d+) - (.*)$"
)


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s).strip()


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class Event:
    ts: datetime
    level: str
    module: str
    func: str
    lineno: int
    kind: str
    is_subagent: bool
    text: str


@dataclass
class AgentNode:
    """A named-agent invocation spawned by a Task → dispatch."""
    task_event: Event
    events: list[Event] = field(default_factory=list)

    @property
    def agent_name(self) -> str:
        m = re.search(r"Task → ([\w\-]+)", self.task_event.text)
        return m.group(1) if m else "?"

    @property
    def title(self) -> str:
        # The text looks like: 🤖 [gladius] Task → agent-name  [TITLE]  desc...
        # We want the LAST [...] that comes after the agent name.
        after = re.sub(r".*Task → [\w\-]+\s*", "", self.task_event.text)
        m = re.search(r"\[([^\]]*)\]", after)
        return m.group(1) if m else ""

    @property
    def ts_start(self) -> datetime:
        return self.task_event.ts

    @property
    def ts_end(self) -> datetime | None:
        for ev in reversed(self.events):
            return ev.ts
        return None

    @property
    def duration_s(self) -> float | None:
        end = self.ts_end
        return (end - self.ts_start).total_seconds() if end else None

    @property
    def n_tools(self) -> int:
        return sum(1 for e in self.events if e.kind == TOOL_USE)

    @property
    def n_errors(self) -> int:
        return sum(1 for e in self.events if e.kind == RESULT_ERR)


@dataclass
class RootNode:
    """The top-level orchestrator run."""
    agent_name: str
    ts_start: datetime | None = None
    ts_end:   datetime | None = None

    # Root-level events before the first Task dispatch (startup reads, etc.)
    preamble: list[Event] = field(default_factory=list)
    # Interleaved list of AgentNode (subtasks) and Event (root-level actions)
    children: list[Union[AgentNode, Event]] = field(default_factory=list)
    # Events after all tasks complete (STATUS, warnings, etc.)
    epilogue: list[Event] = field(default_factory=list)

    @property
    def duration_s(self) -> float | None:
        if self.ts_start and self.ts_end:
            return (self.ts_end - self.ts_start).total_seconds()
        return None

    @property
    def agent_nodes(self) -> list[AgentNode]:
        return [c for c in self.children if isinstance(c, AgentNode)]

    @property
    def n_tasks(self) -> int:
        return len(self.agent_nodes)

    @property
    def n_tools_total(self) -> int:
        root = sum(1 for c in self.children if isinstance(c, Event) and c.kind == TOOL_USE)
        return root + sum(n.n_tools for n in self.agent_nodes)

    @property
    def n_errors_total(self) -> int:
        root = sum(1 for c in self.children if isinstance(c, Event) and c.kind == RESULT_ERR)
        return root + sum(n.n_errors for n in self.agent_nodes)


# ── classification ────────────────────────────────────────────────────────────

def _classify(lineno: int, text: str, module: str, func: str, level: str) -> tuple[str, bool]:
    is_sub = "➣subagent" in text

    if module == "gladius.orchestrator" and "Launching agent" in text:
        return AGENT_LAUNCH, False
    if func == "run_agent" and "▶" in text:
        return AGENT_START, False
    if "🔑" in text and "session" in text:
        return SESSION, False
    if "🤖" in text and "Task →" in text:
        return TASK, False
    if "🧠" in text and "(thinking)" in text:
        return THINKING, is_sub
    if "💬" in text:
        return MESSAGE, is_sub
    if "📋" in text and "TodoWrite" in text:
        return TODO_WRITE, is_sub
    if lineno == 143:
        t = text.strip()
        if t and t[0] in ("✅", "⬜", "🔧"):
            return TODO_ITEM, is_sub
    if "🔧" in text and "[gladius]" in text:
        return TOOL_USE, is_sub
    if lineno == 177:
        t = text.strip()
        if t.startswith("✓"):
            return RESULT_OK, is_sub
        if t.startswith("✗"):
            return RESULT_ERR, is_sub
        return RESULT_CONT, is_sub
    if "━━" in text and "done" in text:
        return STATUS, False
    if level == "WARNING":
        return WARNING, False
    return OTHER, False


# ── raw log → flat event list ─────────────────────────────────────────────────

def _parse_events(content: str) -> list[Event]:
    events: list[Event] = []
    pending: re.Match | None = None
    extra: list[str] = []

    def flush() -> None:
        if pending is None:
            return
        ts_str, level, module, func, lno, msg = pending.groups()
        extra_txt = _strip(" ".join(extra))
        raw = _strip(msg)
        if extra_txt:
            raw = (raw + " " + extra_txt).strip() if raw else extra_txt
        ts     = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        lineno = int(lno)
        kind, is_sub = _classify(lineno, raw, module, func, level)
        events.append(Event(ts, level, module, func, lineno, kind, is_sub, raw))

    for line in content.splitlines():
        m = _LOG_LINE_RE.match(line)
        if m:
            flush()
            pending = m
            extra = []
        else:
            s = line.strip()
            if s and not s.startswith("nohup:"):
                extra.append(line)
    flush()
    return events


# ── flat event list → tree ────────────────────────────────────────────────────

def _build_tree(events: list[Event]) -> RootNode:
    """
    State machine that assembles a RootNode from the flat event list.

    There is no "epilogue phase" — STATUS events occur once per iteration, not
    just at the end of the whole run.  Instead we use two states:

      seen_task=False  — preamble: events before any Task dispatch
      seen_task=True   — events between / inside agent dispatches

    At the end, any trailing non-AgentNode entries in root.children are peeled
    off into root.epilogue so the renderer can give them a distinct label.
    """
    agent_name = "gladius"
    for ev in events:
        if ev.kind == AGENT_LAUNCH:
            m = re.search(r"Launching agent:\s*(.*)", ev.text)
            if m:
                agent_name = m.group(1).strip()
            break

    root = RootNode(
        agent_name=agent_name,
        ts_start=events[0].ts if events else None,
        ts_end=events[-1].ts if events else None,
    )

    current: AgentNode | None = None
    seen_task: bool = False
    last_tool_was_sub: bool = False
    # Buffers root-level events that must appear AFTER the current AgentNode.
    pending_root: list[Event] = []

    def _flush_agent() -> None:
        nonlocal current
        if current is not None:
            root.children.append(current)
            current = None
        root.children.extend(pending_root)
        pending_root.clear()

    for ev in events:
        if ev.kind == OTHER:
            continue

        # Every Task dispatch closes the previous agent and opens a new one.
        if ev.kind == TASK:
            _flush_agent()
            current = AgentNode(task_event=ev)
            last_tool_was_sub = False
            seen_task = True
            continue

        if not seen_task:
            # Preamble: before the very first Task dispatch.
            root.preamble.append(ev)

        elif current is not None:
            # ── Inside an AgentNode ──────────────────────────────────────────
            # TOOL_USE: subagent calls stay in the agent; root calls buffer.
            if ev.kind == TOOL_USE:
                last_tool_was_sub = ev.is_subagent
                if ev.is_subagent:
                    current.events.append(ev)
                else:
                    pending_root.append(ev)
            # RESULT_*: inherit ownership from the preceding TOOL_USE.
            elif ev.kind in (RESULT_OK, RESULT_ERR, RESULT_CONT):
                if last_tool_was_sub:
                    current.events.append(ev)
                else:
                    pending_root.append(ev)
            # Subagent-tagged events (thinking, message…) belong to the agent.
            elif ev.is_subagent:
                current.events.append(ev)
            else:
                # All orchestrator-level events (THINKING, MESSAGE, TODO_*,
                # STATUS, AGENT_START, SESSION, AGENT_LAUNCH, WARNING…)
                # must appear in root.children *after* the AgentNode.
                pending_root.append(ev)

        else:
            # ── Between agents (after STATUS or before next Task) ────────────
            pending_root.append(ev)

    _flush_agent()

    # Peel trailing non-agent events from root.children into root.epilogue so
    # the renderer can give them a distinct "epilogue" label.
    while root.children and isinstance(root.children[-1], Event):
        root.epilogue.insert(0, root.children.pop())

    return root


# ── public API ────────────────────────────────────────────────────────────────

def parse_log(content: str) -> RootNode:
    """Parse a gladius log string into a RootNode tree."""
    return _build_tree(_parse_events(content))
