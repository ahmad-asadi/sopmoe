"""SoP prompt template for the Statement of Performance generation.

Enforces Axioms 1-3 from the paper:
1. Empirical Alignment: scores based on observed drawdown/returns.
2. Conservative Margining: risk penalty for new experts.
3. Tail Separation: highlight extreme-condition performance.
"""

from __future__ import annotations

from typing import Any


def build_sop_prompt(
    timestamp: str,
    market_context: dict[str, Any],
    expert_name: str,
    allocation: list[float] | dict[str, float],
    performance_metrics: dict[str, float],
    uncertainty_estimate: float,
) -> str:
    """Build the prompt for generating a Statement of Performance (SoP).

    Parameters
    ----------
    timestamp :
        ISO-format timestamp for the evaluation point.
    market_context :
        Dict with keys like 'volatility', 'trend', etc.
    expert_name :
        Identifier for the expert being evaluated.
    allocation :
        Portfolio weight vector or dict of asset -> weight.
    performance_metrics :
        Must include 'cumulative_return', 'sharpe', 'max_drawdown'.
    uncertainty_estimate :
        Uncertainty score (0-1).

    Returns
    -------
    Formatted prompt ready to send to the LLM.
    """
    if isinstance(allocation, dict):
        alloc_str = ", ".join(f"{k}: {v:.4f}" for k, v in allocation.items())
    elif isinstance(allocation, (list, tuple)):
        alloc_str = ", ".join(f"{w:.4f}" for w in allocation)
    else:
        alloc_str = str(allocation)

    prompt = f"""[SYSTEM] You are a financial performance analyst. Generate a Statement of Performance (SoP) for the given expert at the given market state. Adhere to:
1. Empirical Alignment: Base scores strictly on observed historical drawdown and returns.
2. Conservative Margining: Apply a risk penalty to new experts.
3. Tail Separation: Highlight performance under extreme conditions.

[CONTEXT]
Timestamp: {timestamp}
Market Volatility: {market_context.get('volatility', 'N/A')}
Market Trend: {market_context.get('trend', 'N/A')}

[EXPERT]
Name: {expert_name}
Allocation: {alloc_str}

[PERFORMANCE]
Cumulative Return: {performance_metrics.get('cumulative_return', 'N/A')}
Sharpe: {performance_metrics.get('sharpe', 'N/A')}
Max Drawdown: {performance_metrics.get('max_drawdown', 'N/A')}

[UNCERTAINTY]
Estimated Uncertainty: {uncertainty_estimate}

Generate a concise SoP (2-3 sentences) summarizing the expert's behavior, risk-adjusted performance, and a confidence score (0-1).

Ensure the output is JSON-parseable with key fields: calculated_rho, confidence_bound_epsilon, regime_separation_margin_tau, tail_optimal_flag."""
    return prompt


def parse_sop_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM response into a structured SoP dict.

    Falls back to default values if parsing fails.
    """
    import json
    import re

    defaults: dict[str, Any] = {
        "calculated_rho": 0.5,
        "confidence_bound_epsilon": 0.1,
        "regime_separation_margin_tau": 0.05,
        "tail_optimal_flag": False,
        "sop_text": "",
    }

    try:
        cleaned = response_text.strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            defaults.update(parsed)
        else:
            defaults["sop_text"] = response_text.strip()
        return defaults
    except (json.JSONDecodeError, ValueError, TypeError):
        defaults["sop_text"] = response_text.strip() or "SoP generation failed"
        return defaults
