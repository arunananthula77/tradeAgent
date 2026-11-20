from src.agents.orchestrator import TradeAgentOrchestrator


def test_end_to_end_mcp_and_agents():
    orch = TradeAgentOrchestrator()
    try:
        tools = orch.mcp.list_tools()
        assert len(tools) == 3

        report = orch.research_ticker("NVDA", days=60)
        assert report["memo"]["ticker"] == "NVDA"
        assert report["memo"]["regime"] in {"Bearish", "Neutral", "Bullish"}
        assert "sharpe_proxy" in report["quant_factors"]
        print("TradeAgent E2E Test Passed:", report["memo"])
    finally:
        orch.close()


if __name__ == "__main__":
    test_end_to_end_mcp_and_agents()
