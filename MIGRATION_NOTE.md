# MIGRATION_NOTE - MCP 2026-07-28 wire - class `header-add`

**Date:** 2026-10-08 - **Lane:** M4 MCP-migration (header-add wave 2, batch 6) - **Branch:** `mcp-2026-wire-header-add`
**Runbook:** `MCP_2026_WIRE_MIGRATION_PLAN_2026-10-07.md` section 3 (header-add) + section 4 (the shim as bridge)
**Deprecation deadline:** the legacy wire dies **2027-07-28** - 12 months after the 2026-07-28 revision.

## 1. Transport reality

Server runs on stdio (`mcp.run()` in `server.py`).

## 2. What changed in this branch

1. `pyproject.toml`: `"mcp>=1.0.0"` -> `"mcp>=2.0.0"` (2.x is the first SDK line speaking
   the 2026-07-28 spec; wave 1 verified `mcp` 2.3.0), plus the packaging include list
   gained the vendored shim so it ships in the wheel.
2. `from mcp.server.fastmcp import FastMCP` -> `from mcp.server.mcpserver import MCPServer as FastMCP` - required, because `mcp.server.fastmcp` raises `ModuleNotFoundError` by design in 2.x (FastMCP was renamed MCPServer). The pin is worthless if the import does not resolve.
3. `mcp2026_shim.py` vendored at the repo root (stdlib, zero third-party deps).
4. `server.py`: migration note + `http_app()` helper = the operator's HTTP-mode enable path
   (`mcp.streamable_http_app(json_response=True)` wrapped in `ShimASGI`).
5. `MIGRATION_NOTE.md`: class, transport reality, steps, verify command, follow-ups.

The shim does the four runbook duties at the transport: read/validate `Mcp-Method` and `Mcp-Name` on ingress, reject a missing `Mcp-Name` on `tools/call` / `resources/read` / `prompts/get` with `-32602`, emit `params._meta.protocolVersion = "2026-07-28"` on every outbound request, and never emit `Mcp-Session-Id` (it strips one if a proxy adds it).

Files changed in this branch: `MIGRATION_NOTE.md`, `mcp2026_shim.py`, `pyproject.toml`, `server.py`.

## 3. Verify

```bash
PYTHONPATH= /opt/homebrew/bin/python3.11 ~/clawd/mcp_wire_audit.py audit --local explainability-report-mcp
```

| state | era | migration |
|---|---|---|
| before (default branch) | 2026-07 | header-add |
| **after (this branch)** | **2026-07** | **handshake-removal** |
| control (migration-note block removed) | 2026-07 | header-add |

The scanner reads the source/manifest files only: it skips `mcp2026_shim.py` by design (`SELF_FILES`) and does not scan `.md`, so neither `MIGRATION_NOTE.md` nor the shim contributes signals above.

**How to read the `after` row honestly.** The audit is a static scan and this tool excludes its own shim from the scan by design (`SELF_FILES`), so `protocol-2026-07-28`, `mcp-method-header`, `mcp-name-header`, `server-discover` and `session-id` in the `after` record are read from the migration-note text, not from executable handshake code. The `session-id` signal in particular is prose (the note documents that the shim *strips* the header) - the control run, which deletes only that note block, drops back to the row above and shows no `session-id` at all. Runtime evidence for the wire is the `mcp>=2.0.0` SDK pin (2.3.0 speaks 2026-07-28) plus the vendored shim at the ingress; `mcp>=2.0.0` alone is not a wire signal for this scanner. **After-rows are note-text-driven until a post-merge re-audit.**

## 4. Follow-ups (not in this branch)

* The *before* `2026-07` era comes with `mcp-name-header` from
  `.well-known/agent-card.json` (`"mcp_name"` registry token) - a known false-positive
  class of this scanner, not wire code. The honest pre-state is the census row.
* Static declaration surfaces (`.well-known/*.json`, `server.json`, `README.md`,
  registry manifests) still declare older wires where applicable - listed as
  follow-ups, not silent-edited (Art. 21: a declaration change gets its own commit).
* stdio carries no HTTP headers: `headers N/A at runtime` until the server is exposed over HTTP, where `ShimASGI` applies. A live probe per plan section 6.5 is owed after merge.

Verify command of record: `PYTHONPATH= /opt/homebrew/bin/python3.11 ~/clawd/mcp_wire_audit.py audit --local <repo>` -> `era: 2026-07`, `migration: none` is the acceptance target for class `header-add`; re-run it after merge, not on this branch's note text.

Plan: `MCP_2026_WIRE_MIGRATION_PLAN_2026-10-07.md` - deadline 2027-07-28 - measurement, not certification.
