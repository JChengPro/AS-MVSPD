"""Stable multi-view paired differential detection."""

from .core import (
    BlockResult,
    JointResult,
    analyze_block,
    analyze_multiview,
    bh_adjust,
    generate_signs,
    self_normalized_feature_scores,
)
from .benchmark import evaluate_published_baselines, simulate_paired_multiview

__all__ = [
    "BlockResult",
    "JointResult",
    "analyze_block",
    "analyze_multiview",
    "bh_adjust",
    "generate_signs",
    "self_normalized_feature_scores",
    "evaluate_published_baselines",
    "simulate_paired_multiview",
]
