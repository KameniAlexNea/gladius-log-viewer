# gladius-log-viewer

A Gradio-based web UI for inspecting **gladius** multi-agent competition runs.

Parses the raw `gladius.log` file into a two-level tree (root orchestrator → subagent tasks) and renders it as an interactive timeline, with collapsible agent blocks, colour-coded event rows, and per-agent stats.

## Features

- **Tree-aware parsing** — correctly separates the root orchestrator from subagent invocations across multiple iterations
- **Multi-iteration support** — each `status=OK` / new `▶ agent started` boundary opens a fresh iteration; events never bleed across iterations
- **Event attribution** — tool calls, results, todo items, and thinking/messages are attributed to the correct agent or root level
- **Consecutive message merging** — adjacent orchestrator messages are collapsed into a single row
- **Collapsible agent blocks** — expand/collapse each `Task →` invocation; header shows agent name, title, duration, tool count, and error count
- **Stats header** — total duration, task count, tool calls, and error count at a glance

## Installation

Requires Python ≥ 3.10.

```bash
uv sync
```

## Usage

```bash
# Launch with a pre-loaded log file
uv run python app.py --log /path/to/gladius.log

# Launch empty (enter path in the UI)
uv run python app.py
```

The UI is served at `http://127.0.0.1:7860` by default. Pass `--host` and `--port` to override.

## Running tests

```bash
uv run pytest
```
