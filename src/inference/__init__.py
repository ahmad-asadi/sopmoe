"""Online inference – retrieval, reasoning & expert switching.

Provides:
- ``InferenceEngine``    – online expert selection pipeline (Algorithm 2)
- ``build_uncertainty_prompt`` – LLM prompt for uncertainty estimation
- ``project_weights``    – simplex projection for portfolio weights
"""

from src.inference.engine import InferenceEngine
from src.inference.prompts import build_uncertainty_prompt
from src.inference.utils import project_weights, similarity_from_distance

__all__ = [
    "InferenceEngine",
    "build_uncertainty_prompt",
    "project_weights",
    "similarity_from_distance",
]
