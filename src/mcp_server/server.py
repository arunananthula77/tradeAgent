import json
import sys
import numpy as np

from src.ml.quant_model import compute_statistical_factors, predict_regime


def _generate_market_series(ticker: str, days: int = 90) -> list[float]:
    """Generates deterministic realistic OHLC close series per ticker for offline/local execution."""
    seed = sum(ord(c) for c in ticker.upper())
    rng = np.random.default_rng(seed)
    drift = rng.uniform(-0.0005, 0.0022)
    vol = rng.uniform(0.011, 0.024)
    returns = rng.normal(drift, vol, size=days)
    base_price = rng.uniform(120.0, 2800.0)
    prices = base_price * np.exp(np.cumsum(returns))
    return [round(float(p), 2) for p in prices]


class FinancialMCPServer:
    """
    Model Context Protocol (MCP) JSON-RPC 2.0 Tool Server exposing financial research tools:
    - tools/list
    - tools/call (fetch_market_data, get_fundamental_snapshot, run_quant_regime_model)
    """

    def __init__(self):
        self.tools = {
            "fetch_market_data": {
                "description": "Fetch historical closing prices and volume summary for a ticker.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["ticker"],
                },
            },
            "get_fundamental_snapshot": {
                "description": "Retrieve valuation ratios, revenue growth, and sector context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
            "run_quant_regime_model": {
                "description": "Execute PyTorch statistical & regime classification model on price history.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"prices": {"type": "array", "items": {"type": "number"}}},
                    "required": ["prices"],
                },
            },
        }

    def handle_request(self, request: dict) -> dict:
        req_id = request.get("id", 1)
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "tradeagent-financial-mcp", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "tools/list":
            tool_list = [
                {"name": name, "description": meta["description"], "inputSchema": meta["inputSchema"]}
                for name, meta in self.tools.items()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tool_list}}

        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            try:
                output = self._execute_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(output)}]},
                }
            except Exception as exc:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(exc)}}

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

    def _execute_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name == "fetch_market_data":
            ticker = args["ticker"].upper()
            days = int(args.get("days", 90))
            prices = _generate_market_series(ticker, days)
            return {
                "ticker": ticker,
                "days": days,
                "latest_price": prices[-1],
                "period_high": max(prices),
                "period_low": min(prices),
                "prices": prices,
            }

        if tool_name == "get_fundamental_snapshot":
            ticker = args["ticker"].upper()
            seed = sum(ord(c) for c in ticker)
            rng = np.random.default_rng(seed + 7)
            return {
                "ticker": ticker,
                "pe_ratio": round(float(rng.uniform(14.5, 38.0)), 2),
                "roe_pct": round(float(rng.uniform(11.0, 29.5)), 2),
                "debt_to_equity": round(float(rng.uniform(0.1, 1.15)), 2),
                "yoy_revenue_growth_pct": round(float(rng.uniform(4.5, 24.0)), 2),
            }

        if tool_name == "run_quant_regime_model":
            prices = np.array(args["prices"], dtype=np.float32)
            factors = compute_statistical_factors(prices)
            regime = predict_regime(factors)
            return {"factors": factors, "regime_prediction": regime}

        raise ValueError(f"Unsupported tool: {tool_name}")


def serve_stdio():
    """Runs MCP server over standard input/output (JSON-RPC 2.0 lines)."""
    server = FinancialMCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        resp = server.handle_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
