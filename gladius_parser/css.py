
# ═══════════════════════════════════════════ CSS ══════════════════════════════

CSS = """<style>
/* ── design tokens ── */
:root {
  --r-bg:       #f8fafc;
  --r-border:   #e2e8f0;
  --r-dim:      #94a3b8;
  --r-text:     #0f172a;

  /* agent colour palette (8) */
  --a0: #3b82f6;  /* blue */
  --a1: #10b981;  /* emerald */
  --a2: #f59e0b;  /* amber */
  --a3: #8b5cf6;  /* violet */
  --a4: #ef4444;  /* red */
  --a5: #06b6d4;  /* cyan */
  --a6: #f97316;  /* orange */
  --a7: #ec4899;  /* pink */
}

/* ── outer wrapper ── */
.lv { font-family: ui-monospace,'SF Mono',Consolas,monospace;
      font-size: 13px; color: var(--r-text); line-height: 1.5; }

/* ── stats header ── */
.lv-header {
  display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
  background: #1e293b; color: #e2e8f0 !important;
  padding: 10px 18px; border-radius: 8px 8px 0 0;
  font-size: 13px;
}
.lv-header-title { font-weight: 700; font-size: 15px; color: #f1f5f9 !important; flex: 1; }
.lv-stat { display: flex; align-items: baseline; gap: 4px; }
.lv-stat-n   { font-weight: 700; color: #f1f5f9 !important; }
.lv-stat-lbl { color: #94a3b8 !important; font-size: 11px; }

/* ── root body ── */
.lv-root {
  border: 1px solid var(--r-border); border-top: none;
  border-radius: 0 0 8px 8px; overflow: hidden;
}

/* ── section labels (preamble / epilogue) ── */
.lv-section-lbl {
  font-size: 10px; font-weight: 700; letter-spacing: .08em;
  color: var(--r-dim); text-transform: uppercase;
  padding: 4px 14px; background: #f1f5f9;
  border-bottom: 1px solid var(--r-border);
}

/* ── agent block ── */
.lv-agent-block {
  border-left: 4px solid var(--agent-color, #3b82f6);
  margin: 0;
}

.lv-agent-header {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 14px;
  background: color-mix(in srgb, var(--agent-color, #3b82f6) 8%, white);
  cursor: pointer; user-select: none;
  border-bottom: 1px solid var(--r-border);
}
.lv-agent-header:hover {
  background: color-mix(in srgb, var(--agent-color, #3b82f6) 14%, white);
}
.lv-agent-name {
  font-weight: 800; font-size: 13px;
  color: var(--agent-color, #3b82f6);
  min-width: 130px;
}
.lv-agent-title {
  font-style: italic; color: #475569; flex: 1;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.lv-agent-meta {
  display: flex; gap: 10px; font-size: 11px; color: var(--r-dim);
  flex-shrink: 0;
}
.lv-agent-meta .err { color: #dc2626; font-weight: 700; }
.lv-toggle { font-size: 11px; color: var(--r-dim); flex-shrink: 0; }

/* agent inner events — shown/hidden via details */
details.lv-agent-details > summary { display: none; }
.lv-agent-events {
  padding: 0;
}

/* ── event row ── */
.lv-ev {
  display: grid;
  grid-template-columns: 64px 22px 1fr;
  gap: 0;
  padding: 4px 10px 4px 14px;
  border-bottom: 1px solid #f1f5f9;
  align-items: start;
}
.lv-ev:last-child { border-bottom: none; }

/* colour bands per kind */
.ev-think { background: #fafafa; }
.ev-msg   { background: #f0f9ff; }
.ev-tool  { background: #fffbeb; }
.ev-tool-s{ background: #fefce8; }
.ev-ok    { background: #f0fdf4; }
.ev-err   { background: #fff1f2; }
.ev-todo  { background: #faf5ff; }
.ev-item  { background: #fdf4ff; padding-left: 28px; }
.ev-session { background: #f8fafc; }
.ev-start   { background: #eef2ff; }
.ev-status  { background: #f0fdfa; }
.ev-warn    { background: #fefce8; }

/* ── cell types ── */
.lv-ts   { font-size: 10px; color: var(--r-dim); padding-top: 2px; white-space: nowrap; }
.lv-icon { font-size: 13px; text-align: center; padding-top: 1px; }
.lv-body { min-width: 0; }

/* ── content helpers ── */
.sub-badge {
  font-size: 10px; background: #fef3c7; color: #92400e;
  border-radius: 3px; padding: 0 4px; font-weight: 700;
  margin-right: 5px; vertical-align: 1px;
}
.tool-name { font-weight: 700; color: #b45309; word-break: break-all; }
.ok-txt    { color: #047857; }
.err-txt   { color: #b91c1c; }
.think-txt { color: #6b7280; font-style: italic; }
.msg-txt   { color: #0369a1; }
.dim       { color: var(--r-dim); font-size: 11px; }

/* badges for tool list */
.tbadges { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 2px; }
.tbadge  { font-size: 10px; background: #ede9fe; color: #5b21b6;
           border-radius: 3px; padding: 1px 5px; }

/* collapsible content */
details.lv-col > summary {
  list-style: none; cursor: pointer; color: var(--r-dim); font-size: 11px; display: inline;
}
details.lv-col > summary::-webkit-details-marker { display: none; }
details.lv-col > summary::before { content: "▶ "; font-size: 9px; }
details.lv-col[open] > summary::before { content: "▼ "; font-size: 9px; }
.lv-pre {
  font-size: 11px; background: #f1f5f9; border-radius: 4px;
  padding: 6px 8px; margin-top: 3px;
  white-space: pre-wrap; word-break: break-all;
  max-height: 260px; overflow-y: auto; overflow-x: hidden;
}
.preview {
  font-size: 11px; color: var(--r-dim);
  max-width: 600px; display: inline-block;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  vertical-align: bottom;
}

/* todo items */
.item-done { color: #047857; }
.item-curr { color: #b45309; }
.item-pend { color: var(--r-dim); }

/* ── empty (retry) agent ── */
.lv-agent-retry {
  border-left: 3px solid #cbd5e1;
  padding: 4px 14px;
  color: #94a3b8;
  font-size: 11px;
  background: #f8fafc;
  border-bottom: 1px solid #f1f5f9;
}
.lv-agent-retry .retry-name {
  font-weight: 700; color: #94a3b8;
}

/* ── iteration tabs ── */
.lv-tab-bar {
  display: flex; gap: 0; padding: 0 8px;
  border-bottom: 2px solid var(--r-border);
  background: #f8fafc;
  overflow-x: auto; flex-wrap: nowrap;
}
.lv-tab-btn {
  padding: 8px 18px;
  font-family: inherit; font-size: 12px; font-weight: 600;
  cursor: pointer; border: none;
  border-bottom: 3px solid transparent; margin-bottom: -2px;
  background: transparent; color: #64748b;
  white-space: nowrap;
}
.lv-tab-btn:hover { color: #1e293b; background: #f1f5f9; }
.lv-tab-active { color: #3b82f6 !important; border-bottom-color: #3b82f6 !important; }
</style>"""


# ═══════════════════════════════════════════ AGENT COLOUR MAP ════════════════

AGENT_COLOURS = [
    "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
    "#ef4444", "#06b6d4", "#f97316", "#ec4899",
]