#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs reports
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="logs/tradeagent_${TIMESTAMP}.log"

TICKERS="${*:-NVDA TCS INFY AAPL}"
echo "[$(date -Iseconds)] Running TradeAgent pipeline for: $TICKERS" | tee -a "$LOG_FILE"

python3 -m src.main --tickers $TICKERS 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] Brief written to reports/latest_brief.md" | tee -a "$LOG_FILE"
