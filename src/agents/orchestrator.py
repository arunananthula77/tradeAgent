import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.agents.skills import SKILL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("TradeAgent.Orchestrator")


class MCPClient:
    """Subprocess JSON-RPC 2.0 client communicating with TradeAgent's MCP Tool Server."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.mcp_server.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self._req_id = 0
        self._send("initialize", {})

    def _send(self, method: str, params: dict) -> dict:
        self._req_id += 1
        payload = {"jsonrpc": "2.0", "id": self._req_id, "method": method, "params": params}
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"MCP Tool Error: {resp['error']}")
        return resp["result"]

    def list_tools(self) -> list[dict]:
        return self._send("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        res = self._send("tools/call", {"name": name, "arguments": arguments})
        text_payload = res["content"][0]["text"]
        return json.loads(text_payload)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=3)


class TradeAgentOrchestrator:
    """
    Multi-agent research orchestrator coordinating Market Data, Fundamental,
    PyTorch Quant, and Risk agents over Model Context Protocol (MCP).
    """

    def __init__(self):
        self.mcp = MCPClient()

    def research_ticker(self, ticker: str, days: int = 90) -> dict:
        start_ts = time.perf_counter()
        ticker = ticker.upper()
        logger.info("Starting multi-agent research workflow for %s", ticker)

        # Agent 1: Market Data Agent
        market_data = self.mcp.call_tool("fetch_market_data", {"ticker": ticker, "days": days})

        # Agent 2: Fundamental Research Agent
        fundamentals = self.mcp.call_tool("get_fundamental_snapshot", {"ticker": ticker})

        # Agent 3: PyTorch Quantitative & Statistical Agent
        quant_output = self.mcp.call_tool(
            "run_quant_regime_model", {"prices": market_data["prices"]}
        )

        context = {
            "ticker": ticker,
            "market": market_data,
            "fundamentals": fundamentals,
            "quant": quant_output,
        }

        # Agent 4: Risk & Thesis Synthesis via Custom Agent Skills
        risk_report = SKILL_REGISTRY["risk_guardrail"].execute(context)
        context["risk"] = risk_report
        memo = SKILL_REGISTRY["investment_memo"].execute(context)

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 2)
        logger.info("Completed research for %s in %sms (Stance: %s)", ticker, elapsed_ms, memo["stance"])

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": elapsed_ms,
            "memo": memo,
            "quant_factors": quant_output["factors"],
            "fundamentals": fundamentals,
        }

    def close(self):
        self.mcp.close()
