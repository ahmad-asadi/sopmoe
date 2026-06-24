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

[OUTPUT STRUCTURE]
You must provide your final answer as a raw JSON object. If you use a thinking process, ensure it is separated from the final JSON output. The JSON must be the only part of the response that follows the thinking process and must be directly parseable by json.loads().

The JSON object must contain exactly these keys:
1. "regime_description": A detailed technical description of the current market regime. Include specific aspects such as volatility levels and trend characteristics (e.g., "High volatility bearish regime with significant downward pressure and erratic price swings").
2. "performance_reasoning": A clear and concise analysis of WHY the model is performing as it is. If performance is poor, analyze the reasons (e.g., "The model's heavy tilt towards momentum assets is causing significant drawdowns in this mean-reverting market"). If performance is good, explain why it's succeeding.
3. "sop_text": A highly formatted, structured summary. Use a clear layout like:
   --- REGIME ANALYSIS ---
   [Your analysis here]
   --- PERFORMANCE EVALUATION ---
   [Your evaluation here]
   --- SUITABILITY VERDICT ---
   [Your verdict here]
4. "calculated_rho": Performance score [0, 1].
5. "confidence_bound_epsilon": Uncertainty bound [0, 1].
6. "regime_separation_margin_tau": Regime separation margin [0, 1].
7. "tail_optimal_flag": Boolean (true/false) indicating if this expert is optimal for tail events.

[GUIDELINES]
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
1. Carefully analyze the provided Market State and Expert Performance Metrics.
2. Determine the specific characteristics of the market regime.
3. Reason through the link between the expert's allocation and its performance in this regime.
4. Generate the final JSON output following the [OUTPUT STRUCTURE] precisely.
"""
    return prompt


def parse_sop_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM response into a structured SoP dict.
    
    Handles responses that may contain thinking process before the final JSON object.
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
        
        # Extract the JSON block. 
        # If there's a thinking process, we look for the last occurrence of a JSON-like structure.
        # We look for the last '{' and the last '}' to isolate the final JSON object.
        start_idx = cleaned.rfind('{')
        end_idx = cleaned.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = cleaned[start_idx : end_idx + 1]
            
            # Handle potential markdown wrapping within the extracted string
            json_str = re.sub(r"^```(?:json)?\n?", "", json_str)
            json_str = re.sub(r"\n?```$", "", json_str)
            
            parsed = json.loads(json_str)
            defaults.update(parsed)
        else:
            defaults["sop_text"] = cleaned
            
        return defaults
    except (json.JSONDecodeError, ValueError, TypeError):
        defaults["sop_text"] = response_text.strip() or "SoP generation failed"
        return defaults

