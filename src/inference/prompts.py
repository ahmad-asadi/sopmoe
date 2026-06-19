"""Uncertainty prompt template for LLM-based uncertainty estimation.

The generated prompt asks the LLM to produce per-expert uncertainty scores
in the range [0, 1], based on the current market embedding and the
retrieved Statements of Performance (SoPs).
"""

from __future__ import annotations

from typing import Any


def build_uncertainty_prompt(
    current_embedding: list[float] | None = None,
    retrieved_records: dict[str, list[dict[str, Any]]] | None = None,
    performance_summaries: dict[str, dict[str, float]] | None = None,
    embedding_dim: int | None = None,
) -> str:
    """Build a prompt that asks the LLM to estimate per-expert uncertainty.

    Parameters
    ----------
    current_embedding :
        The query embedding vector (list of floats).  May be ``None`` if
        only textual context is provided.
    retrieved_records :
        Dict keyed by expert name, where each value is a list of retrieved
        metadata dicts (each containing ``sop_text``, ``cumulative_return``,
        ``sharpe``, ``drawdown``, ``distance``, etc.).
    performance_summaries :
        Dict keyed by expert name with aggregate performance values
        (e.g. ``{"weighted_return": 0.05, "std": 0.02}``).
    embedding_dim :
        Dimension of the embedding (for context).

    Returns
    -------
    str
        Formatted prompt ready to send to the LLM.
    """
    if retrieved_records is None:
        retrieved_records = {}
    if performance_summaries is None:
        performance_summaries = {}

    # Build expert summaries from retrieved records
    expert_sections: list[str] = []
    for expert_name, records in retrieved_records.items():
        sop_excerpts = []
        perf_rows = []
        for rec in records:
            sop = rec.get("sop_text", "")[:200]
            if sop:
                sop_excerpts.append(f"      SoP: {sop}")
            perf_rows.append(
                f"        ret={rec.get('cumulative_return', 0):+.4f}  "
                f"sharpe={rec.get('sharpe', 0):.2f}  "
                f"dd={rec.get('drawdown', 0):.4f}"
            )

        summary = performance_summaries.get(expert_name, {})
        weighted_ret = summary.get("weighted_return", 0.0)
        perf_std = summary.get("std", 0.0)

        section = f"""  Expert: {expert_name}
    Aggregate:
      weighted_return={weighted_ret:+.4f}
      performance_std={perf_std:.4f}
    Retrieved neighbours:
"""
        if sop_excerpts:
            section += "\n".join(sop_excerpts) + "\n"
        else:
            section += "      (No SoP text available)\n"
        if perf_rows:
            section += "\n".join(perf_rows) + "\n"
        expert_sections.append(section)

    experts_str = "\n".join(expert_sections) if expert_sections else "  (No retrieved records)"

    embed_str = f"Embedding dimension: {embedding_dim or 'N/A'}"
    if current_embedding is not None:
        embed_str += f"\nEmbedding (first 8 dims): {current_embedding[:8]}"

    prompt = f"""[SYSTEM] You are an uncertainty estimation module for a regime-aware portfolio selection system. Your task is to output per-expert uncertainty scores U_e in [0, 1] where higher means more uncertain.

Guidelines:
- Experts with consistent performance across retrieved neighbours should receive LOW uncertainty.
- Experts with high variance in performance, contradictory SoPs, or signs of overfitting should receive HIGH uncertainty.
- If an expert has few or no retrieved records, assign uncertainty = 0.5 (neutral).
- Output ONLY valid JSON with the structure:
  {{"expert_uncertainties": {{"<expert_name>": <float 0-1>, ...}}}}

[CONTEXT]
{embed_str}

[RETRIEVED EXPERTS]
{experts_str}

[INSTRUCTION]
Analyse the retrieved historical performance and SoPs for each expert. Consider:
1. Consistency of returns across neighbours
2. Evidence of regime-specific overfitting
3. Divergence between expert behaviour in similar vs. different regimes

Return ONLY a JSON object with per-expert uncertainty scores."""
    return prompt
