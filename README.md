<div align="center">

# Explainability Report MCP

**MCP server for explainability report mcp operations**

[![PyPI](https://img.shields.io/pypi/v/meok-explainability-report-mcp)](https://pypi.org/project/meok-explainability-report-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-MCP_Server-purple)](https://meok.ai)

</div>

## Overview

Explainability Report MCP provides AI-powered tools via the Model Context Protocol (MCP).

## Tools

| Tool | Description |
|------|-------------|
| `quick_scan` | Describe an AI system -> instant transparency and explainability assessment. No  |
| `generate_model_card` | Generate an EU AI Act compliant model card with structured transparency informat |
| `explain_decision` | Generate a human-readable explanation of an AI decision. |
| `transparency_audit` | Assess an AI system against EU AI Act Article 13 transparency requirements. |
| `create_impact_assessment` | Generate a DPIA/AIIA (AI Impact Assessment) template for an AI system. |

## Installation

```bash
pip install meok-explainability-report-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "explainability-report-mcp": {
      "command": "python",
      "args": ["-m", "meok_explainability_report_mcp.server"]
    }
  }
}
```

## Usage with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# This server exposes 5 tool(s) via MCP
# See server.py for full implementation
```

## License

MIT © [MEOK AI Labs](https://meok.ai)
