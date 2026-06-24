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

    prompt = f"""[SYSTEM] You are a high-precision financial performance analyst. Your task is to generate a structured Statement of Performance (SoP) for a specific expert model in a given market regime.

STRICT OUTPUT REQUIREMENT:
You MUST return ONLY a raw JSON object. Do NOT include any thinking process, preamble, explanation, or markdown code blocks (e.g., no ```json ... ```). The response must be directly parseable by json.loads().

GUIDELINES:
1. Empirical Alignment: Base the assessment strictly on observed historical returns and drawdowns.
2. Conservative Margining: Apply a risk penalty to experts with limited track records or high uncertainty.
3. Tail Separation: Explicitly identify if the expert excels or fails under extreme market conditions.

[MARKET STATE]
- Timestamp: {timestamp}
- Volatility: {market_context.get('volatility', 'N/A')}
- Trend: {market_context.get('trend', 'N/A')}

[EXPERT DETAILS]
- Name: {expert_name}
- Portfolio Allocation: {alloc_str}

[PERFORMANCE METRICS]
- Cumulative Return: {performance_metrics.get('cumulative_return', 'N/A')}
- Sharpe Ratio: {performance_metrics.get('sharpe', 'N/A')}
- Max Drawdown: {performance_metrics.get('max_drawdown', 'N/A')}
- Estimated Uncertainty: {uncertainty_estimate}

TASK:
1. Analyze the current market regime based on volatility and trend.
2. Evaluate why the expert's specific allocation is resulting in the observed performance (reasoning).
3. Synthesize this into a structured SoP summary.

OUTPUT FORMAT:
Return a JSON object with these exact keys:
- "regime_description": A concise technical description of the current market regime and its primary characteristics.
- "performance_reasoning": A short analysis of why the model is performing well or poorly in this specific regime (e.g., "The high allocation to BTC is driving returns during this bullish trend").
- "sop_text": A structured summary containing: [Regime Analysis] -> [Performance Evaluation] -> [Suitability Verdict].
- "calculated_rho": Performance score [0, 1].
- "confidence_bound_epsilon": Uncertainty bound [0, 1].
- "regime_separation_margin_tau": Regime separation margin [0, 1].
- "tail_optimal_flag": Boolean indicating if this expert is optimal for tail events.
"""
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
        "regime_description": "N/A",
        "performance_reasoning": "N/A",
    }

    try:
        cleaned = response_text.strip()
        # Remove markdown blocks if present despite instructions
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        
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

