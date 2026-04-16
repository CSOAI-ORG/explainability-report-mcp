# AI Explainability Report MCP Server

By [MEOK AI Labs](https://meok.ai) | The only MCP server for AI explainability and transparency reports.

## Quick Start

```bash
pip install explainability-report-mcp
explainability-report-mcp
```

Or run directly:

```bash
pip install mcp
python server.py
```

## Claude Desktop Config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "explainability-report": {
      "command": "explainability-report-mcp"
    }
  }
}
```

## Tools

| Tool | Description | API Key Required |
|------|-------------|-----------------|
| `quick_scan` | Describe an AI system, get transparency assessment | No |
| `generate_model_card` | Generate EU AI Act compliant model card | No (free tier) |
| `explain_decision` | Generate human-readable explanation of an AI decision | No (free tier) |
| `transparency_audit` | Assess against EU AI Act Article 13 transparency requirements | No (free tier) |
| `create_impact_assessment` | Generate DPIA/AIIA template | No (free tier) |

## Free Tier

10 calls/day per tool, no API key required. Upgrade to Pro ($29/mo) for unlimited access at [meok.ai](https://meok.ai/mcp/explainability-report/pro).

## Examples

### Quick Scan (zero config)
```
quick_scan("Credit scoring model using gradient boosted trees trained on 5 years of loan data")
```

### Generate Model Card
```
generate_model_card("CreditScore-v2", "Predict loan default probability for retail banking customers")
```

### Explain a Decision
```
explain_decision("Loan application denied", "credit_score:620,income:35000,debt_ratio:0.45")
```

## License

MIT - Built by [MEOK AI Labs](https://meok.ai)
