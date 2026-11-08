"""Core statistics for adaptive sparse stable paired differential detection.

The experimental unit is a matched subject. Every randomization multiplies the
complete multi-omics change vector of a subject by the same sign, preserving
cross-view dependence under the paired global null.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class BlockResult:
    name: str
    feature_table: pd.DataFrame
    aggregation_table: pd.DataFrame
    permutation_table: pd.DataFrame
    block_pvalue: float
    selected_k: int
    selected_features: tuple[str, ...]


@dataclass(frozen=True)
class JointResult:
    block_results: Mapping[str, BlockResult]
    joint_table: pd.DataFrame
    joint_pvalue: float
    joint_evidence: float


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def generate_signs(
    n_subjects: int,
    max_random: int = 4095,
    seed: int = 20260803,
) -> tuple[np.ndarray, bool]:
    """Return a closed sign-flip group with the identity pattern first.

    The full group is enumerated when it fits the configured sign-pattern
    budget. Otherwise, a reproducible closed subgroup is enumerated in full.
    Using a subgroup rather than an arbitrary collection of sign patterns
    preserves closure of the randomization orbit, which is required by the
    nested adaptive calibration.

    The boolean return value is ``True`` only for enumeration of the complete
    ``2**n_subjects`` sign-flip group.
    """

    if n_subjects < 2:
        raise ValueError("At least two paired subjects are required")
    if max_random < 1:
        raise ValueError("max_random must allow at least two sign patterns")
    total = 2**n_subjects
    observed = np.ones((1, n_subjects), dtype=float)
    if total <= max_random + 1:
        all_signs = np.asarray(list(product((-1.0, 1.0), repeat=n_subjects)))
        observed_index = np.flatnonzero(np.all(all_signs == 1.0, axis=1))[0]
        all_signs[[0, observed_index]] = all_signs[[observed_index, 0]]
        return all_signs, True

    subgroup_power = min(n_subjects, int(np.floor(np.log2(max_random + 1))))
    if subgroup_power < 1:
        raise ValueError("max_random is too small to construct a sign-flip subgroup")

    rng = np.random.default_rng(seed)
    while True:
        basis = rng.integers(0, 2, size=(subgroup_power, n_subjects), dtype=np.int8)
        if _gf2_rank(basis) == subgroup_power:
            break
    coefficients = np.asarray(
        list(product((0, 1), repeat=subgroup_power)), dtype=np.int8
    )
    binary_patterns = (coefficients @ basis) % 2
    subgroup = 1.0 - 2.0 * binary_patterns.astype(float)
    return subgroup, False


def _gf2_rank(matrix: np.ndarray) -> int:
    """Return matrix rank over the two-element finite field."""

    values = np.asarray(matrix, dtype=np.int8).copy() % 2
    n_rows, n_cols = values.shape
    rank = 0
    for column in range(n_cols):
        pivots = np.flatnonzero(values[rank:, column])
        if len(pivots) == 0:
            continue
        pivot = rank + int(pivots[0])
        values[[rank, pivot]] = values[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and values[row, column]:
                values[row] ^= values[rank]
        rank += 1
        if rank == n_rows:
            break
    return rank


def _tail_probabilities(values: np.ndarray) -> np.ndarray:
    """Permutation-orbit upper-tail probabilities, retaining ties."""

    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)
    first_not_below = np.searchsorted(ordered, values - EPS, side="left")
    return (len(values) - first_not_below) / len(values)


def self_normalized_feature_scores(
    delta: np.ndarray,
    signs: np.ndarray,
    influence_constraint: bool = False,
) -> np.ndarray:
    """Calculate feature evidence for every row-wise sign pattern.

    The main AS-MVSPD score is |sum(d_i)| / sqrt(sum(d_i^2)). The optional
    leave-one-subject constraint is retained only for ablation: it takes the
    minimum aligned score across the complete sample and every leave-one fit,
    and assigns zero when any leave-one fit reverses sign.
    """

    values = np.asarray(delta, dtype=float)
    sign_patterns = np.asarray(signs, dtype=float)
    if values.ndim != 2 or sign_patterns.ndim != 2:
        raise ValueError("delta and signs must both be two-dimensional")
    if values.shape[0] != sign_patterns.shape[1]:
        raise ValueError("Sign patterns do not match the number of subjects")
    if not np.isfinite(values).all():
        raise ValueError("delta contains non-finite values")

    sum_squares = np.sum(values * values, axis=0)
    valid = sum_squares > EPS
    signed_sums = sign_patterns @ values
    denominator = np.sqrt(np.maximum(sum_squares, EPS))
    full_score = np.abs(signed_sums) / denominator
    full_score[:, ~valid] = 0.0
    if not influence_constraint:
        return full_score

    direction = np.sign(signed_sums)
    robust = full_score.copy()
    for subject_index in range(values.shape[0]):
        loo_sum = signed_sums - (
            sign_patterns[:, subject_index, None] * values[subject_index, None, :]
        )
        loo_ss = sum_squares - values[subject_index] ** 2
        loo_denominator = np.sqrt(np.maximum(loo_ss, EPS))
        aligned = direction * loo_sum / loo_denominator
        aligned[:, loo_ss <= EPS] = 0.0
        robust = np.minimum(robust, aligned)
    return np.maximum(robust, 0.0)


def _top_k_means(scores: np.ndarray, k_grid: Sequence[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    n_features = scores.shape[1]
    grid = tuple(sorted({max(1, min(int(k), n_features)) for k in k_grid}))
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    statistics = np.column_stack([cumulative[:, k - 1] / k for k in grid])
    return statistics, grid


def analyze_block(
    delta: pd.DataFrame,
    signs: np.ndarray,
    name: str,
    k_grid: Sequence[int] = (1, 2, 5, 10, 20),
    influence_constraint: bool = False,
) -> BlockResult:
    if delta.shape[0] != signs.shape[1]:
        raise ValueError(f"{name}: sample count does not match sign matrix")
    if delta.shape[1] == 0:
        raise ValueError(f"{name}: no features")

    feature_scores = self_normalized_feature_scores(
        delta.to_numpy(dtype=float),
        signs,
        influence_constraint=influence_constraint,
    )
    aggregated, grid = _top_k_means(feature_scores, k_grid)
    scale_p = np.column_stack(
        [_tail_probabilities(aggregated[:, index]) for index in range(len(grid))]
    )
    scale_evidence = -np.log(np.maximum(scale_p, 1.0 / len(signs)))
    adaptive_evidence = scale_evidence.max(axis=1)
    block_p = _tail_probabilities(adaptive_evidence)
    selected_grid_index = int(np.argmax(scale_evidence[0]))
    selected_k = grid[selected_grid_index]

    observed_scores = feature_scores[0]
    selected_indices = np.argsort(-observed_scores, kind="stable")[:selected_k]
    selected_features = tuple(str(delta.columns[index]) for index in selected_indices)
    feature_p = np.mean(
        feature_scores >= observed_scores[None, :] - EPS,
        axis=0,
    )
    max_scores = feature_scores.max(axis=1)
    feature_fwer = np.mean(
        max_scores[:, None] >= observed_scores[None, :] - EPS,
        axis=0,
    )
    values = delta.to_numpy(dtype=float)
    means = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    dz = np.divide(means, sd, out=np.zeros_like(means), where=sd > EPS)
    direction_n = np.maximum((values > 0).sum(axis=0), (values < 0).sum(axis=0))
    feature_table = pd.DataFrame(
        {
            "feature": delta.columns.astype(str),
            "mean_delta": means,
            "sd_delta": sd,
            "cohen_dz": dz,
            "direction_consistent_n": direction_n,
            "direction_consistency": direction_n / len(delta),
            "stable_score": observed_scores,
            "exact_feature_p": feature_p,
            "feature_fdr": bh_adjust(feature_p),
            "maxT_fwer": feature_fwer,
            "selected": False,
        }
    )
    feature_table.loc[selected_indices, "selected"] = True
    feature_table = feature_table.sort_values(
        ["stable_score", "cohen_dz", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    feature_table["stable_rank"] = np.arange(1, len(feature_table) + 1)

    aggregation_table = pd.DataFrame(
        {
            "k": grid,
            "observed_top_k_mean": aggregated[0],
            "scale_exact_p": scale_p[0],
            "scale_evidence": scale_evidence[0],
            "selected_scale": [index == selected_grid_index for index in range(len(grid))],
        }
    )
    permutation_table = pd.DataFrame(
        {
            "permutation_index": np.arange(len(signs)),
            "adaptive_evidence": adaptive_evidence,
            "block_exact_p_rank": block_p,
        }
    )
    for index, k in enumerate(grid):
        permutation_table[f"top_{k}_mean"] = aggregated[:, index]
        permutation_table[f"top_{k}_tail_p"] = scale_p[:, index]

    return BlockResult(
        name=name,
        feature_table=feature_table,
        aggregation_table=aggregation_table,
        permutation_table=permutation_table,
        block_pvalue=float(block_p[0]),
        selected_k=selected_k,
        selected_features=selected_features,
    )


def analyze_multiview(
    blocks: Mapping[str, pd.DataFrame],
    signs: np.ndarray | None = None,
    k_grid: Sequence[int] = (1, 2, 5, 10, 20),
    influence_constraint: bool = False,
    max_random: int = 4095,
    seed: int = 20260803,
) -> JointResult:
    if len(blocks) < 2:
        raise ValueError("At least two omics blocks are required")
    first = next(iter(blocks.values()))
    if any(not first.index.equals(block.index) for block in blocks.values()):
        raise ValueError("All omics blocks must have identical subject order")
    if signs is None:
        signs, _ = generate_signs(len(first), max_random=max_random, seed=seed)

    block_results = {
        name: analyze_block(
            block,
            signs,
            name=name,
            k_grid=k_grid,
            influence_constraint=influence_constraint,
        )
        for name, block in blocks.items()
    }
    block_p_matrix = np.column_stack(
        [result.permutation_table["block_exact_p_rank"] for result in block_results.values()]
    )
    block_evidence = -np.log(np.maximum(block_p_matrix, 1.0 / len(signs)))
    joint_evidence = block_evidence.min(axis=1)
    # Intersection-union test: the alternative requires a shift in every block.
    # Therefore the valid combined p-value is the largest component p-value.
    joint_p = block_p_matrix.max(axis=1)
    joint_table = pd.DataFrame(
        {
            "permutation_index": np.arange(len(signs)),
            "joint_evidence": joint_evidence,
            "joint_iut_pvalue": joint_p,
        }
    )
    for index, name in enumerate(block_results):
        joint_table[f"{name}_block_p_rank"] = block_p_matrix[:, index]

    return JointResult(
        block_results=block_results,
        joint_table=joint_table,
        joint_pvalue=float(joint_p[0]),
        joint_evidence=float(joint_evidence[0]),
    )
