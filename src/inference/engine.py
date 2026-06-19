"""Online inference engine implementing Algorithm 2 – expert selection via
similarity-based retrieval, uncertainty estimation, and utility maximisation.

For each test-time step:
  1. Get embedding ``h_t`` from the VAE.
  2. Retrieve top‑K similar historical regimes from the index.
  3. Aggregate per-expert performance (similarity-weighted average).
  4. Estimate per-expert uncertainty via LLM (option A) or heuristic std (option B).
  5. Compute risk-adjusted utility ``U_t^e = R_hat_t^e - λ * U_t^e(uncertainty)``.
  6. Select the expert with highest utility and return its projected weights.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import torch

from src.embedding.vae import DualStreamVAE
from src.experts.registry import ExpertRegistry
from src.indexing.database import VectorDatabase
from src.inference.prompts import build_uncertainty_prompt
from src.inference.utils import project_weights, similarity_from_distance
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceEngine:
    """Online expert selection via retrieval-augmented LLM-guided switching.

    Parameters
    ----------
    vae :
        Trained ``DualStreamVAE`` in evaluation mode.
    expert_registry :
        Registry with all candidate experts.
    db :
        ``VectorDatabase`` (FAISS + SQLite) populated by the offline index.
    llm_client :
        Optional callable ``llm_client(prompt: str) -> str``.  If ``None``
        or if the call fails, a heuristic fallback is used for uncertainty.
    config :
        Dict-like config with keys:
        - ``K`` (int, default 5) – number of neighbours to retrieve.
        - ``lambda_`` (float, default 0.1) – risk-penalty coefficient.
        - ``device`` (str, default "cpu") – torch device.
        - ``performance_metric`` (str, default "cumulative_return") –
          which metric to use for ``R_tau^e``.
        - ``cache_size`` (int, default 100) – max entries in embedding
          result cache.
    """

    def __init__(
        self,
        vae: DualStreamVAE,
        expert_registry: ExpertRegistry,
        db: VectorDatabase,
        llm_client: Callable[[str], str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.vae = vae
        self.expert_registry = expert_registry
        self.db = db
        self.llm_client = llm_client

        config = config or {}
        self.K = int(config.get("K", 5))
        self.lambda_ = float(config.get("lambda_", 0.1))
        self.device = str(config.get("device", "cpu"))
        self.performance_metric = str(config.get("performance_metric", "cumulative_return"))

        self.vae = self.vae.to(self.device)
        self.vae.eval()

        # LRU cache: embedding_hash -> (expert_name, projected_weights, metadata)
        self._cache: OrderedDict[str, tuple[str, np.ndarray, dict[str, Any]]] = OrderedDict()
        self._cache_size = int(config.get("cache_size", 100))

        self._expert_names = self.expert_registry.list_experts()
        logger.info(
            "InferenceEngine initialised: K=%d, λ=%.3f, device=%s, experts=%s",
            self.K,
            self.lambda_,
            self.device,
            self._expert_names,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def select_expert(
        self,
        tech_state: torch.Tensor,
        mkt_state: torch.Tensor,
        timestamp: str | None = None,
        expert_state: np.ndarray | None = None,
    ) -> tuple[str, np.ndarray, dict[str, Any]]:
        """Run the full inference pipeline for a single timestep.

        Parameters
        ----------
        tech_state :
            Technical state tensor of shape ``(L, A * F1)`` or ``(L, A, F1)``.
        mkt_state :
            Market state tensor of shape ``(L, F2)``.
        timestamp :
            Optional ISO timestamp (used for logging only).
        expert_state :
            Optional 1-D state vector for the final ``get_weights`` call.
            If ``None``, the method will still return the selected expert's
            name and fallback weights (uniform).

        Returns
        -------
        expert_name : str
            Name of the selected expert.
        projected_weights : np.ndarray
            Feasible portfolio weights (sum to 1).
        metadata : dict
            Debug info including per-expert utility scores.
        """
        # 1. Get embedding from VAE
        embedding = self._compute_embedding(tech_state, mkt_state)
        cache_key = self._hash_embedding(embedding)

        # Check cache
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            logger.debug("Cache hit for embedding %s", cache_key[:12])
            return cached

        # 2. Retrieve top-K neighbours per expert
        retrieved = self._retrieve_per_expert(embedding)

        # 3. Compute similarity-weighted average performance per expert
        weighted_returns, metadata_per_expert = self._aggregate_performance(retrieved)

        # 4. Estimate uncertainty (LLM or heuristic)
        uncertainties = self._estimate_uncertainties(
            embedding=embedding,
            retrieved=retrieved,
            weighted_returns=weighted_returns,
        )

        # 5. Compute utility = weighted_return - lambda * uncertainty
        utilities: dict[str, float] = {}
        for expert_name in self._expert_names:
            R = weighted_returns.get(expert_name, 0.0)
            U = uncertainties.get(expert_name, 0.5)
            utilities[expert_name] = R - self.lambda_ * U

        # 6. Select best expert
        best_expert = max(utilities, key=utilities.get)
        best_utility = utilities[best_expert]

        # Get weights from the selected expert
        if expert_state is not None:
            expert = self.expert_registry.get_expert(best_expert)
            raw_weights = expert.get_weights(expert_state)
        else:
            n_assets = len(self.expert_registry.list_experts())
            raw_weights = np.ones(n_assets + 1, dtype=np.float32) / (n_assets + 1)

        projected = project_weights(raw_weights)

        metadata: dict[str, Any] = {
            "selected_expert": best_expert,
            "utility": float(best_utility),
            "utilities": {k: float(v) for k, v in utilities.items()},
            "weighted_returns": {k: float(v) for k, v in weighted_returns.items()},
            "uncertainties": {k: float(v) for k, v in uncertainties.items()},
            "neighbour_counts": {
                k: len(v) for k, v in retrieved.items()
            },
            "timestamp": timestamp,
        }

        # Update cache
        self._cache[cache_key] = (best_expert, projected, metadata)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        logger.info(
            "Selected expert '%s' (U=%.4f) from %d candidates",
            best_expert, best_utility, len(self._expert_names),
        )

        return best_expert, projected, metadata

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _compute_embedding(
        self, tech_state: torch.Tensor, mkt_state: torch.Tensor
    ) -> np.ndarray:
        if tech_state.dim() == 3:
            B, L, A_F1 = tech_state.shape
            tech_flat = tech_state.reshape(B, L * A_F1).to(self.device)
        elif tech_state.dim() == 2:
            tech_flat = tech_state.unsqueeze(0).to(self.device)
        else:
            tech_flat = tech_state.to(self.device)

        if mkt_state.dim() == 2:
            mkt_input = mkt_state.unsqueeze(0).to(self.device)
        else:
            mkt_input = mkt_state.to(self.device)

        h = self.vae.get_embedding(tech_flat, mkt_input)
        return h.cpu().numpy().astype(np.float32).flatten()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_per_expert(
        self, embedding: np.ndarray
    ) -> dict[str, list[dict[str, Any]]]:
        """Query the index with ``embedding``, returning top-K per expert."""
        retrieved: dict[str, list[dict[str, Any]]] = {}
        for expert_name in self._expert_names:
            results = self.db.query_similar(
                embedding=embedding,
                k=self.K,
                expert_id=expert_name,
            )
            retrieved[expert_name] = results
        return retrieved

    # ------------------------------------------------------------------
    # Performance aggregation
    # ------------------------------------------------------------------

    def _aggregate_performance(
        self, retrieved: dict[str, list[dict[str, Any]]]
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        """Compute similarity-weighted average of the chosen metric.

        Returns
        -------
        weighted_returns : dict[str, float]
            Per-expert weighted average performance.
        metadata : dict[str, dict[str, Any]]
            Per-expert debug info (mean, std, neighbour count, etc.).
        """
        weighted_returns: dict[str, float] = {}
        per_expert_meta: dict[str, dict[str, Any]] = {}

        for expert_name, records in retrieved.items():
            if not records:
                weighted_returns[expert_name] = 0.0
                per_expert_meta[expert_name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "weighted_return": 0.0,
                    "n_neighbours": 0,
                }
                continue

            distances = np.array([r.get("distance", 1.0) for r in records], dtype=np.float64)
            performances = np.array(
                [r.get(self.performance_metric, 0.0) for r in records],
                dtype=np.float64,
            )
            similarities = similarity_from_distance(distances)

            weighted_avg = float(np.sum(similarities * performances))
            mean_val = float(np.mean(performances))
            std_val = float(np.std(performances)) if len(performances) > 1 else 0.0

            weighted_returns[expert_name] = weighted_avg
            per_expert_meta[expert_name] = {
                "mean": mean_val,
                "std": std_val,
                "weighted_return": weighted_avg,
                "n_neighbours": len(records),
            }

        return weighted_returns, per_expert_meta

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    def _estimate_uncertainties(
        self,
        embedding: np.ndarray,
        retrieved: dict[str, list[dict[str, Any]]],
        weighted_returns: dict[str, float],
    ) -> dict[str, float]:
        """Estimate per-expert uncertainty.

        Option A: LLM-based estimation via a single batched prompt.
        Option B (fallback): standard deviation of retrieved performances.
        """
        # Build performance summaries
        perf_summaries: dict[str, dict[str, float]] = {}
        for expert_name, records in retrieved.items():
            if records:
                perfs = [r.get(self.performance_metric, 0.0) for r in records]
                std_val = float(np.std(perfs)) if len(perfs) > 1 else 0.0
            else:
                std_val = 0.0
            perf_summaries[expert_name] = {
                "weighted_return": weighted_returns.get(expert_name, 0.0),
                "std": std_val,
            }

        # Option A: LLM call
        if self.llm_client is not None:
            try:
                return self._llm_uncertainty(embedding, retrieved, perf_summaries)
            except Exception as exc:
                logger.warning("LLM uncertainty estimation failed: %s – falling back", exc)

        # Option B: Heuristic (std of retrieved performances)
        return self._heuristic_uncertainty(retrieved, perf_summaries)

    def _llm_uncertainty(
        self,
        embedding: np.ndarray,
        retrieved: dict[str, list[dict[str, Any]]],
        perf_summaries: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Call the LLM to obtain per-expert uncertainty scores."""
        prompt = build_uncertainty_prompt(
            current_embedding=embedding.tolist() if embedding is not None else None,
            retrieved_records=retrieved,
            performance_summaries=perf_summaries,
            embedding_dim=len(embedding) if embedding is not None else None,
        )

        response = self.llm_client(prompt)
        parsed = self._parse_uncertainty_response(response)

        # Fill in defaults for any expert missing from the LLM response
        uncertainties: dict[str, float] = {}
        for expert_name in self._expert_names:
            val = parsed.get(expert_name)
            if val is not None and 0.0 <= val <= 1.0:
                uncertainties[expert_name] = val
            else:
                # Fall back to heuristic for this expert
                records = retrieved.get(expert_name, [])
                if records:
                    perfs = [r.get(self.performance_metric, 0.0) for r in records]
                    uncertainties[expert_name] = min(float(np.std(perfs)) * 2.0, 1.0)
                else:
                    uncertainties[expert_name] = 0.5

        return uncertainties

    def _heuristic_uncertainty(
        self,
        retrieved: dict[str, list[dict[str, Any]]],
        perf_summaries: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Heuristic uncertainty: normalised std of retrieved performances.

        Maps performance std to [0, 1] via ``min(std * 2.0, 1.0)``.
        """
        uncertainties: dict[str, float] = {}
        for expert_name in self._expert_names:
            records = retrieved.get(expert_name, [])
            if not records:
                uncertainties[expert_name] = 0.5
            else:
                perfs = [r.get(self.performance_metric, 0.0) for r in records]
                std_val = float(np.std(perfs)) if len(perfs) > 1 else 0.0
                uncertainties[expert_name] = min(std_val * 2.0, 1.0)
        return uncertainties

    @staticmethod
    def _parse_uncertainty_response(response: str) -> dict[str, float]:
        """Parse the LLM response JSON into a dict of expert -> uncertainty."""
        import re
        try:
            cleaned = response.strip()
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                raw = data.get("expert_uncertainties", data)
                if isinstance(raw, dict):
                    return {str(k): float(v) for k, v in raw.items()}
            return {}
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse LLM uncertainty response")
            return {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_embedding(embedding: np.ndarray) -> str:
        return hashlib.sha256(embedding.tobytes()).hexdigest()

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.debug("InferenceEngine cache cleared")
