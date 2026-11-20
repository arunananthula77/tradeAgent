#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 - << 'PY'
from src.agents.orchestrator import TradeAgentOrchestrator

orch = TradeAgentOrchestrator()
tools = [t["name"] for t in orch.mcp.list_tools()]
orch.close()
assert set(tools) == {"fetch_market_data", "get_fundamental_snapshot", "run_quant_regime_model"}
print("HEALTHCHECK OK - Registered MCP tools:", tools)
PY
