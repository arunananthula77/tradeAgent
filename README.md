# TradeAgent — Multi-Agent Financial Research System (MCP + PyTorch)

An automated multi-agent financial research pipeline built with **Python**, **PyTorch**, **Model Context Protocol (MCP)**, and **Linux Bash automation** to eliminate manual research bottlenecks for small trading teams.

## Architecture
- **MCP Tool Server (`src/mcp_server/server.py`)**: JSON-RPC 2.0 server over stdio exposing structured financial tools (`fetch_market_data`, `get_fundamental_snapshot`, `run_quant_regime_model`).
- **PyTorch Quantitative Engine (`src/ml/quant_model.py`)**: Computes statistical factors (annualized volatility, Sharpe proxy, 14-day RSI, trailing max drawdown, 20-day momentum) and classifies market regimes (`Bearish`, `Neutral`, `Bullish`) via a PyTorch neural network.
- **Multi-Agent Orchestrator & Custom Skills (`src/agents/`)**: Coordinates tool calls across specialized agents and executes composable domain skills (`risk_guardrail`, `investment_memo`).
- **Linux Automation & Monitoring (`scripts/`)**: Bash entrypoints (`run_research.sh`, `health_check.sh`) with timestamped execution logs and automated Markdown/JSON brief exports.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run health check & end-to-end tests
chmod +x scripts/*.sh
./scripts/health_check.sh
python3 -m tests.test_tradeagent

# 3. Run multi-agent research workflow
./scripts/run_research.sh NVDA TCS INFY AAPL
```
# tradeAgent
