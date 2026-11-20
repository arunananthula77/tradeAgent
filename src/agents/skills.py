from dataclasses import dataclass
from typing import Callable


@dataclass
class AgentSkill:
    """Reusable domain skill definition invoked by TradeAgent orchestrator."""
    name: str
    description: str
    required_tools: list[str]
    execute: Callable[[dict], dict]


def evaluate_risk_skill(context: dict) -> dict:
    """Custom Skill: Synthesizes volatility, drawdown, and debt ratios into an actionable risk grade."""
    factors = context["quant"]["factors"]
    fundamentals = context["fundamentals"]

    vol = factors["annualized_volatility"]
    max_dd = abs(factors["max_drawdown"])
    dte = fundamentals["debt_to_equity"]

    risk_score = 0
    flags = []
    if vol > 0.30:
        risk_score += 2
        flags.append(f"Elevated annualized volatility ({vol:.1%})")
    if max_dd > 0.15:
        risk_score += 2
        flags.append(f"Deep trailing drawdown ({max_dd:.1%})")
    if dte > 0.85:
        risk_score += 1
        flags.append(f"Higher leverage (D/E {dte})")

    if risk_score >= 3:
        rating = "HIGH"
    elif risk_score >= 1:
        rating = "MODERATE"
    else:
        rating = "LOW"

    return {
        "risk_rating": rating,
        "risk_score": risk_score,
        "flags": flags if flags else ["Within normal volatility and leverage bounds"],
    }


def generate_thesis_skill(context: dict) -> dict:
    """Custom Skill: Combines quantitative regime, fundamentals, and risk rating into a trade memo."""
    ticker = context["ticker"]
    regime_info = context["quant"]["regime_prediction"]
    factors = context["quant"]["factors"]
    fundamentals = context["fundamentals"]
    risk = context["risk"]

    regime = regime_info["regime"]
    if regime == "Bullish" and risk["risk_rating"] != "HIGH":
        stance = "ACCUMULATE / LONG BIAS"
    elif regime == "Bearish" or risk["risk_rating"] == "HIGH":
        stance = "DEFENSIVE / HEDGE"
    else:
        stance = "HOLD / WATCHLIST"

    return {
        "ticker": ticker,
        "stance": stance,
        "regime": regime,
        "confidence": regime_info["confidence"],
        "key_metrics": {
            "latest_price": context["market"]["latest_price"],
            "sharpe_proxy": factors["sharpe_proxy"],
            "rsi_14": factors["rsi_14"],
            "pe_ratio": fundamentals["pe_ratio"],
            "yoy_revenue_growth_pct": fundamentals["yoy_revenue_growth_pct"],
        },
        "risk_assessment": risk,
    }


SKILL_REGISTRY = {
    "risk_guardrail": AgentSkill(
        name="risk_guardrail",
        description="Evaluates statistical volatility, max drawdown, and balance-sheet leverage.",
        required_tools=["fetch_market_data", "get_fundamental_snapshot", "run_quant_regime_model"],
        execute=evaluate_risk_skill,
    ),
    "investment_memo": AgentSkill(
        name="investment_memo",
        description="Produces structured multi-factor research brief for trading teams.",
        required_tools=["fetch_market_data", "get_fundamental_snapshot", "run_quant_regime_model"],
        execute=generate_thesis_skill,
    ),
}
