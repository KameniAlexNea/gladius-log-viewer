"""Gladius log parser — tree model.

The log represents a two-level agent hierarchy:

    gladius (root orchestrator)
    └── does its own tool calls / thinking / messages
    └── Agent → <agent-name>  ← opens a child AgentNode
             └── subagent tool calls / results
    └── resumes: more root-level events
    └── Agent → <next-agent>
             └── ...

Log format (new)
----------------
    TIMESTAMP LEVEL [it=N at=N agent=NAME] message

    Where LEVEL is INFO/DEBUG/WARNING and agent=NAME tells us which agent
    generated the line.  Events are classified solely from emoji / keyword
    patterns in the message text; agent attribution uses the agent header field.

    Subagent events have agent != root-agent-name (e.g. agent=scout).
    Root events have agent == root-agent-name (e.g. agent=gladius).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Union

# ── event kinds ───────────────────────────────────────────────────────────────

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

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# New log format: TIMESTAMP LEVEL [it=X at=X agent=NAME] message
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(\w+)\s+"
    r"\[it=([^\s]+)\s+at=([^\s]+)\s+agent=([^\]]+)\](.*)"
)


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s).strip()


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class Event:
    ts: datetime
    level: str
    it: str
    at: str
    agent: str
    kind: str
    is_subagent: bool
    text: str


@dataclass
class AgentNode:
    """A named-agent invocation spawned by a Task/Agent dispatch."""
    task_event: Event
    events: list[Event] = field(default_factory=list)

    @property
    def agent_name(self) -> str:
        m = re.search(r"(?:Task|Agent) \u2192 (\S+)", self.task_event.text)
        return m.group(1) if m else "?"

    @property
    def title(self) -> str:
        # The text looks like: 🤖 [gladius] Task/Agent → agent-name  [TITLE] ...
        # We want the LAST [...] that comes after the agent name.
        after = re.sub(r".*(?:Task|Agent) \u2192 \S+\s*", "", self.task_event.text)
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

def _classify(text: str, level: str) -> str:
    """Classify the event kind from message content alone."""
    t = text.strip()
    if not t:
        return OTHER
    if level == "WARNING":
        return WARNING
    if "🤖" in t and ("Task \u2192" in t or "Agent \u2192" in t):
        return TASK
    if "▶" in t:
        return AGENT_START
    if "🔑" in t and "session" in t:
        return SESSION
    if "🧠" in t and "(thinking)" in t:
        return THINKING
    if "💬" in t:
        return MESSAGE
    if "📋" in t and "TodoWrite" in t:
        return TODO_WRITE
    if "🔧" in t and "[gladius]" in t:
        return TOOL_USE
    # TODO_ITEM: status emoji without a [gladius] prefix (not a task-completion line)
    if t.startswith(("✅", "⬜", "🔧")) and "[gladius]" not in t:
        return TODO_ITEM
    if t.startswith("✓"):
        return RESULT_OK
    if t.startswith("✗"):
        return RESULT_ERR
    if "━━" in t and "done" in t:
        return STATUS
    return OTHER


# ── raw log → flat event list ─────────────────────────────────────────────────

def _parse_events(content: str) -> list[Event]:
    """Parse every log line into an Event; detect root-agent from first AGENT_START."""
    # Collect raw (ts_str, level, it, at, agent, msg) tuples first so we can
    # determine the root-agent name before computing is_subagent.
    raw: list[tuple[str, str, str, str, str, str]] = []
    for line in content.splitlines():
        s = _strip(line)
        if not s or s.startswith("nohup:"):
            continue
        m = _LOG_LINE_RE.match(s)
        if not m:
            continue
        ts_str, level, it, at, agent, msg = m.groups()
        msg = msg.strip()
        if msg:
            raw.append((ts_str, level, it, at, agent, msg))

    # Determine root agent (first line whose message contains ▶, i.e. AGENT_START).
    root_agent = "gladius"
    for _, _, _, _, agent, msg in raw:
        if "▶" in msg:
            root_agent = agent
            break

    non_root = {"-", "orchestrator", root_agent}
    events: list[Event] = []
    in_result = False  # True after RESULT_OK/ERR until next non-continuation event

    for ts_str, level, it, at, agent, msg in raw:
        ts   = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        kind = _classify(msg, level)

        # Tool-output continuation: any unrecognised line immediately following a
        # result, from a named agent, that isn't a hook/progress line.
        if (
            kind == OTHER
            and in_result
            and not msg.startswith("[gladius]")
            and "⏳" not in msg
            and "🚀" not in msg
        ):
            kind = RESULT_CONT

        if kind in (RESULT_OK, RESULT_ERR):
            in_result = True
        elif kind != RESULT_CONT:
            in_result = False

        is_sub = agent not in non_root
        events.append(Event(
            ts=ts, level=level, it=it, at=at, agent=agent,
            kind=kind, is_subagent=is_sub, text=msg,
        ))

    return events


# ── flat event list → tree ────────────────────────────────────────────────────

def _build_tree(events: list[Event]) -> RootNode:
    """
    Assemble a RootNode from the flat event list.

    is_subagent is already set correctly on every Event (agent header field),
    so routing is simple: subagent events → current AgentNode; root events →
    pending_root buffer (flushed into root.children after the agent closes).

    States:
      seen_task=False  — preamble: before the first Task dispatch
      seen_task=True   — inside or between agent dispatches

    Trailing non-AgentNode entries in root.children are peeled into
    root.epilogue so the renderer can give them a distinct label.
    """
    # Root-agent name from the first AGENT_START event.
    agent_name = "gladius"
    for ev in events:
        if ev.kind == AGENT_START:
            agent_name = ev.agent
            break

    root = RootNode(
        agent_name=agent_name,
        ts_start=events[0].ts if events else None,
        ts_end=events[-1].ts if events else None,
    )

    current: AgentNode | None = None
    seen_task: bool = False
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

        if ev.kind == TASK:
            _flush_agent()
            current = AgentNode(task_event=ev)
            seen_task = True
            continue

        if not seen_task:
            root.preamble.append(ev)

        elif current is not None:
            # is_subagent is authoritative: route to agent or pending_root.
            if ev.is_subagent:
                current.events.append(ev)
            else:
                pending_root.append(ev)

        else:
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
