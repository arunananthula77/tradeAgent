import argparse
import json
import os
from src.agents.orchestrator import TradeAgentOrchestrator


def format_markdown_brief(reports: list[dict]) -> str:
    lines = [
        "# TradeAgent Automated Multi-Agent Research Brief",
        "",
        "| Ticker | Stance | Regime (Conf) | Price | Sharpe | RSI-14 | P/E | YoY Growth | Risk |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for item in reports:
        m = item["memo"]
        km = m["key_metrics"]
        risk = m["risk_assessment"]["risk_rating"]
        lines.append(
            f"| **{m['ticker']}** | {m['stance']} | {m['regime']} ({m['confidence']:.0%}) "
            f"| {km['latest_price']:.2f} | {km['sharpe_proxy']:.2f} | {km['rsi_14']:.1f} "
            f"| {km['pe_ratio']:.1f}x | {km['yoy_revenue_growth_pct']:.1f}% | {risk} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TradeAgent: Multi-Agent Financial Research via MCP & PyTorch")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["NVDA", "TCS", "INFY", "AAPL"],
        help="List of equity tickers to analyze",
    )
    parser.add_argument("--days", type=int, default=90, help="Trailing lookback window in days")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)
    orchestrator = TradeAgentOrchestrator()
    try:
        all_reports = [orchestrator.research_ticker(t, days=args.days) for t in args.tickers]
    finally:
        orchestrator.close()

    md_output = format_markdown_brief(all_reports)
    print("\n" + md_output)

    with open("reports/latest_brief.md", "w", encoding="utf-8") as f:
        f.write(md_output)
    with open("reports/latest_brief.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)


if __name__ == "__main__":
    main()
