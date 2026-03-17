"""Tests for the gladius_parser tree-building logic.

Each test constructs a minimal synthetic log and asserts the exact shape of
the resulting RootNode, covering:

  - preamble / epilogue attribution
  - ordering: root events appear AFTER the agent they follow
  - subagent results stay inside their AgentNode
  - root-level (non-subagent) tool calls and results go to root.children
  - iteration boundary: AGENT_START / SESSION are placed at root, not absorbed
  - multi-iteration: agents from different iterations are distinct
  - subagent thinking/message stays in agent, not leaking to root
  - counters: n_tools, n_errors, n_tasks, n_tools_total, n_errors_total
"""

from __future__ import annotations

import pytest

from gladius_parser.parser import (
    AGENT_LAUNCH, AGENT_START, MESSAGE, RESULT_CONT, RESULT_ERR, RESULT_OK,
    SESSION, STATUS, TASK, THINKING, TODO_ITEM, TODO_WRITE, TOOL_USE, WARNING,
    AgentNode, Event, RootNode, parse_log,
)

# ── minimal log-line factory ──────────────────────────────────────────────────

_D = "2026-01-01"


def _line(t: str, level: str, module: str, func: str, lno: int, msg: str) -> str:
    return f"{_D} {t} | {level:<8} | {module}:{func}:{lno} - {msg}"


def mk_agent_start(t="00:00:01.000", tools="Read"):
    return _line(t, "DEBUG", "gladius.roles.agent_runner", "run_agent", 96,
                 f"  \u25b6 [gladius]  tools=['{tools}']")


def mk_session(t="00:00:02.000", sid="sess1"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_message", 90,
                 f"  \U0001f511 [gladius] session={sid}")


def mk_task(t: str, agent: str, title: str = "work"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_use", 158,
                 f"  \U0001f916 [gladius] Task \u2192 {agent}  [{title}]  description")


def mk_sub_tool(t: str, name: str = "Read"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_use", 165,
                 f'  \U0001f527 [gladius] \u27a3subagent {name}  {{"arg": "val"}}')


def mk_root_tool(t: str, name: str = "Read"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_use", 165,
                 f'  \U0001f527 [gladius] {name}  {{"arg": "val"}}')


def mk_result_ok(t: str, text: str = "ok result"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_result", 177,
                 f"  \u2713 [gladius] {text}")


def mk_result_err(t: str, text: str = "error"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_result", 177,
                 f"  \u2717 [gladius] {text}")


def mk_result_cont(t: str, text: str = "2\u2192continuation"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_result", 177,
                 f"       {text}")


def mk_thinking(t: str, text: str = "thinking", sub: bool = False):
    badge = " \u27a3subagent" if sub else ""
    return _line(t, "DEBUG", "gladius.roles._console", "_log_message", 106,
                 f"  \U0001f9e0 [gladius]{badge} (thinking) {text}")


def mk_message(t: str, text: str = "hello", sub: bool = False):
    badge = " \u27a3subagent" if sub else ""
    return _line(t, "DEBUG", "gladius.roles._console", "_log_message", 102,
                 f"  \U0001f4ac [gladius]{badge} {text}")


def mk_todo_write(t: str, n: int = 3, sub: bool = False):
    badge = " \u27a3subagent" if sub else ""
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_use", 136,
                 f"  \U0001f4cb [gladius]{badge} TodoWrite  0/{n} done")


def mk_todo_item(t: str, text: str = "do it", done: bool = False):
    icon = "\u2705" if done else "\u2b1c"
    return _line(t, "DEBUG", "gladius.roles._console", "_log_tool_use", 143,
                 f"       {icon}  {text}")


def mk_status(t: str = "23:59:59.000"):
    return _line(t, "DEBUG", "gladius.roles._console", "_log_status", 122,
                 "  \u2501\u2501\u2501\u2501\u2501 [gladius] done  status=OK")


# ── tree-shape helpers ────────────────────────────────────────────────────────

def agents(root: RootNode) -> list[AgentNode]:
    return [c for c in root.children if isinstance(c, AgentNode)]


def child_kinds(root: RootNode) -> list[str]:
    """Compact label per child: 'agent:<name>' or event.kind."""
    out = []
    for c in root.children:
        if isinstance(c, AgentNode):
            out.append(f"agent:{c.agent_name}")
        else:
            out.append(c.kind)
    return out


def log(*lines: str) -> str:
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Preamble / epilogue
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreamble:
    def test_agent_start_and_session_before_first_task_go_to_preamble(self):
        root = parse_log(log(
            mk_agent_start("00:00:01.000"),
            mk_session("00:00:02.000"),
            mk_task("00:01:00.000", "alpha"),
            mk_status(),
        ))
        assert len(root.preamble) == 2
        assert root.preamble[0].kind == AGENT_START
        assert root.preamble[1].kind == SESSION

    def test_no_preamble_when_task_is_first(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "alpha"),
            mk_status(),
        ))
        assert root.preamble == []

    def test_status_goes_to_epilogue(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "alpha"),
            mk_status("00:02:00.000"),
        ))
        assert len(root.epilogue) == 1
        assert root.epilogue[0].kind == STATUS


# ═══════════════════════════════════════════════════════════════════════════════
# Basic single-agent structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicTree:
    def test_single_agent_name_and_title(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "team-lead", "plan baseline"),
            mk_status(),
        ))
        node = agents(root)[0]
        assert node.agent_name == "team-lead"
        assert node.title == "plan baseline"

    def test_subagent_events_collected_in_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "alpha"),
            mk_sub_tool("00:01:10.000", "Read"),
            mk_result_ok("00:01:11.000", "file contents"),
            mk_result_cont("00:01:11.001", "2\u2192more"),
            mk_status(),
        ))
        node = agents(root)[0]
        assert len(node.events) == 3  # tool_use + result_ok + result_cont
        assert node.events[0].kind == TOOL_USE
        assert node.events[1].kind == RESULT_OK
        assert node.events[2].kind == RESULT_CONT


# ═══════════════════════════════════════════════════════════════════════════════
# Ordering: root events must appear AFTER the agent they follow
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrdering:
    def test_root_thinking_after_agent_is_placed_after_agent(self):
        """
        TASK(a) → sub-tool → result-ok → root-thinking → TASK(b) → STATUS
        children: [agent:a, thinking, agent:b]   NOT [thinking, agent:a, agent:b]
        """
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_thinking("00:01:20.000", "now dispatch b"),   # root orchestrator
            mk_task("00:01:21.000", "agent-b"),
            mk_status(),
        ))
        assert child_kinds(root) == ["agent:agent-a", THINKING, "agent:agent-b"]

    def test_root_message_after_agent_is_placed_after_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_message("00:01:20.000", "transitioning"),
            mk_task("00:01:21.000", "agent-b"),
            mk_status(),
        ))
        assert child_kinds(root) == ["agent:agent-a", MESSAGE, "agent:agent-b"]

    def test_multiple_root_events_between_agents_preserve_order(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_thinking("00:01:20.000"),
            mk_message("00:01:21.000"),
            mk_todo_write("00:01:22.000"),
            mk_task("00:01:23.000", "agent-b"),
            mk_status(),
        ))
        assert child_kinds(root) == [
            "agent:agent-a", THINKING, MESSAGE, TODO_WRITE, "agent:agent-b"
        ]

    def test_root_todo_items_after_agent_are_placed_correctly(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_todo_write("00:01:20.000"),
            mk_todo_item("00:01:20.001", "step 1"),
            mk_todo_item("00:01:20.002", "step 2"),
            mk_task("00:01:21.000", "agent-b"),
            mk_status(),
        ))
        kinds = child_kinds(root)
        assert kinds[0] == "agent:agent-a"
        assert kinds[-1] == "agent:agent-b"
        # todo_write and two todo_items appear between the agents
        assert TODO_WRITE in kinds
        assert kinds.count(TODO_ITEM) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Subagent result attribution: results must stay inside the AgentNode
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubagentResultAttribution:
    def test_subagent_ok_results_stay_in_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000", "first line"),
            mk_result_cont("00:01:11.001", "2\u2192second"),
            mk_result_cont("00:01:11.002", "3\u2192third"),
            mk_task("00:01:20.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        assert node_a.events[0].kind == TOOL_USE
        assert node_a.events[1].kind == RESULT_OK
        assert node_a.events[2].kind == RESULT_CONT
        assert node_a.events[3].kind == RESULT_CONT
        assert len(node_a.events) == 4

    def test_subagent_err_results_stay_in_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000", "Bash"),
            mk_result_err("00:01:11.000", "Exit code 1"),
            mk_result_cont("00:01:11.001", "Traceback..."),
            mk_task("00:01:20.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        assert node_a.n_errors == 1
        assert len(node_a.events) == 3

    def test_subagent_results_do_not_appear_in_root_children(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_result_cont("00:01:11.001"),
            mk_task("00:01:20.000", "agent-b"),
            mk_status(),
        ))
        ev_kinds = {c.kind for c in root.children if isinstance(c, Event)}
        assert RESULT_OK not in ev_kinds
        assert RESULT_CONT not in ev_kinds

    def test_multiple_subagent_tools_results_all_stay_in_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000", "Read"),
            mk_result_ok("00:01:11.000"),
            mk_sub_tool("00:01:12.000", "Bash"),
            mk_result_err("00:01:13.000"),
            mk_sub_tool("00:01:14.000", "Write"),
            mk_result_ok("00:01:15.000"),
            mk_task("00:01:20.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        assert node_a.n_tools == 3
        assert node_a.n_errors == 1
        # All six events (3 tools + 3 results) in the agent
        assert len(node_a.events) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# Root-level (non-subagent) tool calls go to root.children, not agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestRootToolCalls:
    def test_root_tool_and_result_appear_in_root_children(self):
        """Orchestrator reads a file directly (no ➣subagent): tool + result go to root."""
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),        # subagent tool → in agent
            mk_result_ok("00:01:11.000"),        # → in agent
            mk_root_tool("00:01:20.000"),        # root tool → in root
            mk_result_ok("00:01:21.000", "root result"),  # → in root
            mk_task("00:01:30.000", "agent-b"),
            mk_status(),
        ))
        kinds = child_kinds(root)
        assert kinds == ["agent:agent-a", TOOL_USE, RESULT_OK, "agent:agent-b"]

    def test_root_tool_not_absorbed_into_previous_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_root_tool("00:01:20.000"),
            mk_result_ok("00:01:21.000"),
            mk_task("00:01:30.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        # agent-a only has the subagent tool + its result
        root_tools_in_agent = [e for e in node_a.events
                                if e.kind == TOOL_USE and not e.is_subagent]
        assert root_tools_in_agent == []
        assert len(node_a.events) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Iteration boundary: AGENT_START / SESSION must NOT be swallowed into prev agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestIterationBoundary:
    def test_agent_start_at_boundary_goes_to_root_children(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "ml-engineer"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_thinking("00:01:20.000", "iter 1 done"),
            mk_message("00:01:21.000", "summary"),
            mk_agent_start("00:01:22.000"),   # new iteration
            mk_session("00:01:23.000", "sess2"),
            mk_task("00:01:24.000", "team-lead"),
            mk_status(),
        ))
        assert child_kinds(root) == [
            "agent:ml-engineer",
            THINKING,
            MESSAGE,
            AGENT_START,
            SESSION,
            "agent:team-lead",
        ]

    def test_agent_start_not_in_previous_agent_events(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "ml-engineer"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_agent_start("00:01:22.000"),
            mk_session("00:01:23.000"),
            mk_task("00:01:24.000", "team-lead"),
            mk_status(),
        ))
        ml = agents(root)[0]
        assert all(e.kind not in (AGENT_START, SESSION) for e in ml.events)

    def test_session_not_in_previous_agent_events(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_session("00:01:20.000", "new-session"),
            mk_task("00:01:21.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        assert all(e.kind != SESSION for e in node_a.events)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-iteration: agents from different iterations must be separate
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiIteration:
    def test_two_iterations_produce_distinct_agent_nodes(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "ml-engineer", "iter1"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_thinking("00:01:20.000", "iter 1 complete"),
            mk_agent_start("00:01:21.000"),
            mk_session("00:01:22.000", "sess2"),
            mk_task("00:01:23.000", "ml-engineer", "iter2"),
            mk_sub_tool("00:01:30.000"),
            mk_result_ok("00:01:31.000"),
            mk_status(),
        ))
        ml_agents = [n for n in agents(root) if n.agent_name == "ml-engineer"]
        assert len(ml_agents) == 2
        assert ml_agents[0].title == "iter1"
        assert ml_agents[1].title == "iter2"

    def test_iter2_events_not_in_iter1_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "ml-engineer", "iter1"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_agent_start("00:01:20.000"),
            mk_task("00:01:21.000", "ml-engineer", "iter2"),
            mk_sub_tool("00:01:30.000"),
            mk_result_ok("00:01:31.000"),
            mk_status(),
        ))
        iter1, iter2 = [n for n in agents(root) if n.agent_name == "ml-engineer"]
        assert len(iter1.events) == 2   # only its own tool + result
        assert len(iter2.events) == 2

    def test_children_order_across_two_iterations(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "ml-engineer", "iter1"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_thinking("00:01:20.000", "done iter1"),
            mk_agent_start("00:01:21.000"),
            mk_session("00:01:22.000"),
            mk_task("00:01:23.000", "ml-engineer", "iter2"),
            mk_status(),
        ))
        assert child_kinds(root) == [
            "agent:ml-engineer",   # iter1
            THINKING,
            AGENT_START,
            SESSION,
            "agent:ml-engineer",   # iter2
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Subagent thinking / message stays inside the agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubagentOwnEvents:
    def test_subagent_thinking_stays_in_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_thinking("00:01:05.000", "sub thought", sub=True),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_task("00:01:20.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        assert any(e.kind == THINKING for e in node_a.events)
        assert THINKING not in child_kinds(root)

    def test_subagent_message_stays_in_agent(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_message("00:01:05.000", "sub says hi", sub=True),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_task("00:01:20.000", "agent-b"),
            mk_status(),
        ))
        node_a = agents(root)[0]
        assert any(e.kind == MESSAGE for e in node_a.events)
        # Root children must not contain a stray message
        assert MESSAGE not in child_kinds(root)


# ═══════════════════════════════════════════════════════════════════════════════
# Counters
# ═══════════════════════════════════════════════════════════════════════════════

class TestCounters:
    def test_n_tools_and_n_errors(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_ok("00:01:11.000"),
            mk_sub_tool("00:01:12.000"),
            mk_result_err("00:01:13.000"),
            mk_status(),
        ))
        node = agents(root)[0]
        assert node.n_tools == 2
        assert node.n_errors == 1

    def test_n_tasks(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "a"),
            mk_task("00:01:10.000", "b"),
            mk_task("00:01:20.000", "c"),
            mk_status(),
        ))
        assert root.n_tasks == 3

    def test_n_tools_total_includes_root_tools(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),    # agent tool
            mk_result_ok("00:01:11.000"),
            mk_root_tool("00:01:20.000"),   # root tool (between agents)
            mk_result_ok("00:01:21.000"),
            mk_task("00:01:30.000", "agent-b"),
            mk_status(),
        ))
        # 1 subagent tool + 1 root tool = 2 total
        assert root.n_tools_total == 2

    def test_n_errors_total_includes_root_errors(self):
        root = parse_log(log(
            mk_task("00:01:00.000", "agent-a"),
            mk_sub_tool("00:01:10.000"),
            mk_result_err("00:01:11.000"),    # agent error
            mk_root_tool("00:01:20.000"),
            mk_result_err("00:01:21.000"),    # root error
            mk_task("00:01:30.000", "agent-b"),
            mk_status(),
        ))
        assert root.n_errors_total == 2
