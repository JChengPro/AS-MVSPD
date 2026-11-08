"""Benchmark helpers for stable paired multi-view detection."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .core import (
    EPS,
    _tail_probabilities,
    _top_k_means,
    analyze_multiview,
    self_normalized_feature_scores,
)


def fast_stable_multiview_pvalue(
    blocks: Mapping[str, pd.DataFrame],
    signs: np.ndarray,
    k_grid: Sequence[int] = (1, 2, 5, 10, 20),
    influence_constraint: bool = False,
) -> float:
    """Return the AS-MVSPD joint p-value without building result tables."""

    block_pvalues = []
    for block in blocks.values():
        scores = self_normalized_feature_scores(
            block.to_numpy(dtype=float), signs, influence_constraint
        )
        aggregated, grid = _top_k_means(scores, k_grid)
        scale_p = np.column_stack(
            [_tail_probabilities(aggregated[:, index]) for index in range(len(grid))]
        )
        adaptive = (-np.log(np.maximum(scale_p, 1.0 / len(signs)))).max(axis=1)
        block_pvalues.append(float(_tail_probabilities(adaptive)[0]))
    return float(max(block_pvalues))


def fast_maxT_pvalues(
    block: pd.DataFrame,
    signs: np.ndarray,
    influence_constraint: bool = False,
) -> np.ndarray:
    """Return single-step maxT adjusted p-values without result-table overhead."""

    scores = self_normalized_feature_scores(
        block.to_numpy(dtype=float), signs, influence_constraint
    )
    observed = scores[0]
    maxima = scores.max(axis=1)
    return np.mean(maxima[:, None] >= observed[None, :] - EPS, axis=0)


def _paired_cq_block_pvalue(values: np.ndarray, signs: np.ndarray) -> float:
    """Sign-flip calibration of the one-sample Chen-Qin quadratic statistic."""

    statistic = _paired_cq_statistics(values, signs)
    return float(_tail_probabilities(statistic)[0])


def _paired_cq_statistics(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Return the paired one-sample Chen-Qin U-statistic over a sign orbit."""

    n_subjects = values.shape[0]
    if n_subjects < 2:
        raise ValueError("The Chen-Qin statistic requires at least two subjects")
    signed_sums = signs @ values
    constant = float(np.sum(values * values))
    return (np.sum(signed_sums * signed_sums, axis=1) - constant) / (
        n_subjects * (n_subjects - 1)
    )


def _paired_aspu_block_pvalue(
    values: np.ndarray,
    signs: np.ndarray,
    powers: Sequence[float] = (1, 2, 3, 4, 5, 6, np.inf),
) -> float:
    """Paired sign-flip adaptation of the adaptive sum-of-powers test.

    Feature scores are self-normalized signed sums. Odd powers use a two-sided
    absolute aggregate, even powers use their non-negative aggregate, and the
    infinite-power statistic is the maximum absolute score. Selection of the
    power is recalibrated over the same closed sign-flip orbit.
    """

    adaptive_evidence = _paired_aspu_adaptive_evidence(values, signs, powers)
    return float(_tail_probabilities(adaptive_evidence)[0])


def _paired_aspu_adaptive_evidence(
    values: np.ndarray,
    signs: np.ndarray,
    powers: Sequence[float] = (1, 2, 3, 4, 5, 6, np.inf),
) -> np.ndarray:
    """Return orbit-recalibrated adaptive sum-of-powers evidence."""

    denominator = np.sqrt(np.maximum(np.sum(values * values, axis=0), EPS))
    scores = (signs @ values) / denominator
    statistics = []
    for power in powers:
        if np.isinf(power):
            statistic = np.max(np.abs(scores), axis=1)
        else:
            raw = np.sum(scores ** int(power), axis=1)
            statistic = np.abs(raw) if int(power) % 2 else raw
        statistics.append(statistic)
    scale_p = np.column_stack([_tail_probabilities(x) for x in statistics])
    return -np.log(np.maximum(scale_p.min(axis=1), 1.0 / len(signs)))


def _signed_rank_scores(values: np.ndarray) -> np.ndarray:
    """Return per-feature signed ranks, excluding exact zero differences."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional paired-difference matrix")
    nonzero = np.abs(values) > EPS
    ranks = rankdata(np.abs(values), axis=0, method="average")
    # rankdata includes zero differences at the bottom of each feature. Remove
    # their count so non-zero observations retain the ranks they would receive
    # after the standard Wilcoxon zero-difference exclusion.
    zero_counts = (~nonzero).sum(axis=0)
    ranks = np.where(nonzero, ranks - zero_counts[None, :], 0.0)
    return np.sign(values) * ranks


def _paired_rankcom_adaptive_evidence(
    values: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Orbit-calibrated signed-rank max/sum Cauchy-combination evidence.

    Zhang and Feng's one-sample method combines maximum- and sum-type
    Wilcoxon signed-rank statistics. Here the same two rank components are
    recalibrated on the closed paired sign orbit, and their Cauchy combination
    is ranked once more on that orbit. This preserves finite-sample
    comparability with the other paired multi-view baselines.
    """

    signed_ranks = _signed_rank_scores(values)
    denominator = np.sqrt(np.maximum(np.sum(signed_ranks * signed_ranks, axis=0), EPS))
    standardized = (signs @ signed_ranks) / denominator
    max_statistic = np.max(np.abs(standardized), axis=1)
    sum_statistic = np.mean(standardized * standardized, axis=1)
    component_p = np.column_stack(
        [
            _tail_probabilities(max_statistic),
            _tail_probabilities(sum_statistic),
        ]
    )
    orbit_size = len(signs)
    clipped = np.clip(component_p, 0.5 / orbit_size, 1.0 - 0.5 / orbit_size)
    cauchy_statistic = np.mean(np.tan((0.5 - clipped) * np.pi), axis=1)
    combined_p = 0.5 - np.arctan(cauchy_statistic) / np.pi
    return -np.log(np.maximum(combined_p, 1.0 / orbit_size))


def _paired_rankcom_block_pvalue(values: np.ndarray, signs: np.ndarray) -> float:
    """Paired closed-orbit P value for the rank-combination baseline."""

    evidence = _paired_rankcom_adaptive_evidence(values, signs)
    return float(_tail_probabilities(evidence)[0])


def evaluate_published_baselines(
    blocks: Mapping[str, pd.DataFrame],
    signs: np.ndarray,
) -> dict[str, float]:
    """Evaluate published high-dimensional mean-test adaptations fairly.

    All baselines operate on paired differences and use the same subject-wise
    sign-flip group as AS-MVSPD. View-level p-values are combined by the same
    intersection-union rule, so differences reflect the block statistic rather
    than a different multi-view null hypothesis.
    """

    cq = []
    aspu = []
    rankcom = []
    for block in blocks.values():
        values = block.to_numpy(dtype=float)
        cq.append(_paired_cq_block_pvalue(values, signs))
        aspu.append(_paired_aspu_block_pvalue(values, signs))
        rankcom.append(_paired_rankcom_block_pvalue(values, signs))
    return {
        "Paired-aSPU-IUT": float(max(aspu)),
        "Paired-CQ-IUT": float(max(cq)),
        "Paired-RankCOM-IUT": float(max(rankcom)),
    }


def simple_multiview_pvalue(
    blocks: Mapping[str, pd.DataFrame],
    signs: np.ndarray,
    aggregation: str,
) -> float:
    """Calibrated conjunction test for non-adaptive benchmark statistics."""

    block_pvalues = []
    for block in blocks.values():
        scores = self_normalized_feature_scores(
            block.to_numpy(dtype=float), signs, influence_constraint=False
        )
        if aggregation == "max":
            statistic = scores.max(axis=1)
        elif aggregation == "dense":
            statistic = np.mean(scores * scores, axis=1)
        else:
            raise ValueError(aggregation)
        block_pvalues.append(_tail_probabilities(statistic))
    block_p = np.column_stack(block_pvalues)
    return float(block_p[0].max())


def evaluate_methods(
    blocks: Mapping[str, pd.DataFrame],
    signs: np.ndarray,
    k_grid: tuple[int, ...] = (1, 2, 5, 10, 20),
) -> tuple[dict[str, float], dict[str, dict[str, set[str]]]]:
    loo_variant = analyze_multiview(
        blocks,
        signs=signs,
        k_grid=k_grid,
        influence_constraint=True,
    )
    ordinary = analyze_multiview(
        blocks,
        signs=signs,
        k_grid=k_grid,
        influence_constraint=False,
    )
    pvalues = {
        "AS-MVSPD": ordinary.joint_pvalue,
        "AS-MVSPD+LOO": loo_variant.joint_pvalue,
        "Max-MV": simple_multiview_pvalue(blocks, signs, "max"),
        "Dense-MV": simple_multiview_pvalue(blocks, signs, "dense"),
    }
    selections = {
        "AS-MVSPD": {
            name: set(result.selected_features)
            for name, result in ordinary.block_results.items()
        },
        "AS-MVSPD+LOO": {
            name: set(result.selected_features)
            for name, result in loo_variant.block_results.items()
        },
    }
    return pvalues, selections


def simulate_paired_blocks(
    n_subjects: int,
    n_features: int,
    n_signal: int,
    scenario: str,
    rng: np.random.Generator,
    effect: float = 1.4,
    loading_scale: float = 0.35,
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    """Generate two high-dimensional blocks with shared subject-level noise."""

    normalized_scenario = "partial_views" if scenario == "one_block" else scenario
    active_views = (0,) if scenario == "one_block" else None
    blocks, truth, _ = simulate_paired_multiview(
        n_subjects=n_subjects,
        n_features=n_features,
        n_signal=n_signal,
        scenario=normalized_scenario,
        rng=rng,
        effect=effect,
        loading_scale=loading_scale,
        n_views=2,
        active_views=active_views,
    )
    return blocks, truth


def structured_correlation_factor(
    n_features: int,
    structure: str,
    rho: float,
    block_size: int = 20,
) -> np.ndarray:
    """Return a Cholesky factor for an interpretable feature covariance.

    ``ar1`` uses Sigma[j, l] = rho ** |j-l|. ``block`` uses compound
    symmetry within non-overlapping blocks and zero correlation between
    blocks. Both parameterizations have unit marginal variance.
    """

    if n_features < 1:
        raise ValueError("n_features must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must satisfy 0 <= rho < 1")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    if structure == "ar1":
        indices = np.arange(n_features)
        covariance = rho ** np.abs(indices[:, None] - indices[None, :])
    elif structure == "block":
        covariance = np.eye(n_features)
        for start in range(0, n_features, block_size):
            stop = min(start + block_size, n_features)
            covariance[start:stop, start:stop] = rho
            covariance[np.arange(start, stop), np.arange(start, stop)] = 1.0
    else:
        raise ValueError(f"Unknown correlation structure: {structure}")
    return np.linalg.cholesky(covariance)


def simulate_structured_correlated_multiview(
    n_subjects: int,
    n_features: int,
    n_signal: int,
    scenario: str,
    rng: np.random.Generator,
    structure: str,
    rho: float,
    effect: float = 1.4,
    n_views: int = 3,
    cross_view_correlation: float = 0.3,
    block_size: int = 20,
    signal_directions: Sequence[float] | None = None,
    active_views: Sequence[int] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    """Simulate paired changes with controlled within- and cross-view correlation.

    Every view has the same AR(1) or block-diagonal feature covariance. A
    shared Gaussian component induces the requested correlation between
    corresponding null features in different views. Stable shifts can have
    different signs across views because the multi-view alternative concerns
    the presence, rather than the direction, of a location change.
    """

    if n_subjects < 2 or n_views < 2:
        raise ValueError("At least two subjects and two views are required")
    if not 0 <= n_signal <= n_features:
        raise ValueError("n_signal must be between zero and n_features")
    if scenario not in {"null", "partial_views", "stable_sparse"}:
        raise ValueError(f"Unknown structured simulation scenario: {scenario}")
    if not 0.0 <= cross_view_correlation < 1.0:
        raise ValueError("cross_view_correlation must satisfy 0 <= r < 1")

    if signal_directions is None:
        directions = np.ones(n_views, dtype=float)
    else:
        directions = np.asarray(signal_directions, dtype=float)
        if directions.shape != (n_views,) or np.any(directions == 0):
            raise ValueError("signal_directions must contain one non-zero value per view")
        directions = np.sign(directions)
    if active_views is None:
        active = set(range(n_views)) if scenario == "stable_sparse" else set()
    else:
        active = {int(value) for value in active_views}
        if not active.issubset(set(range(n_views))):
            raise ValueError("active_views contains an invalid view index")
    if scenario == "partial_views" and not active:
        raise ValueError("partial_views requires at least one active view")

    factor = structured_correlation_factor(
        n_features=n_features,
        structure=structure,
        rho=rho,
        block_size=block_size,
    )
    shared = rng.normal(size=(n_subjects, n_features)) @ factor.T
    shared_scale = np.sqrt(cross_view_correlation)
    private_scale = np.sqrt(1.0 - cross_view_correlation)
    signal_indices = np.arange(n_signal)

    blocks: dict[str, pd.DataFrame] = {}
    truth: dict[str, set[str]] = {}
    for view_index in range(n_views):
        name = f"View{view_index + 1}"
        private = rng.normal(size=(n_subjects, n_features)) @ factor.T
        values = shared_scale * shared + private_scale * private
        features = [f"{name}_f{index:04d}" for index in range(n_features)]
        if view_index in active and n_signal:
            values[:, signal_indices] += effect * directions[view_index]
            truth[name] = {features[index] for index in signal_indices}
        else:
            truth[name] = set()
        blocks[name] = pd.DataFrame(values, columns=features)
    return blocks, truth


def simulate_paired_multiview(
    n_subjects: int,
    n_features: int | Sequence[int],
    n_signal: int,
    scenario: str,
    rng: np.random.Generator,
    effect: float = 1.4,
    loading_scale: float = 0.35,
    n_views: int = 3,
    active_views: Sequence[int] | None = None,
    noise_distribution: str = "normal",
    contamination_amplitude: float | None = None,
    n_contamination: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]], dict[str, set[str]]]:
    """Generate paired high-dimensional blocks for method validation.

    ``active_views`` identifies views receiving a stable location shift. The
    contamination scenarios add a one-subject shift to separate null features,
    allowing direct measurement of single-subject-driven false selections.
    """

    if isinstance(n_features, int):
        feature_counts = [int(n_features)] * n_views
    else:
        feature_counts = [int(x) for x in n_features]
        if len(feature_counts) != n_views:
            raise ValueError("n_features must provide one value per view")
    if active_views is None:
        active = set(range(n_views))
    else:
        active = {int(x) for x in active_views}
    if not active.issubset(set(range(n_views))):
        raise ValueError("active_views contains an invalid view index")

    shared = _draw_noise(rng, (n_subjects, 3), noise_distribution)
    
    blocks: dict[str, pd.DataFrame] = {}
    truth: dict[str, set[str]] = {}
    contaminants: dict[str, set[str]] = {}
    for block_index in range(n_views):
        block_name = f"View{block_index + 1}"
        block_features = feature_counts[block_index]
        loadings = rng.normal(scale=loading_scale, size=(3, block_features))
        noise = _draw_noise(rng, (n_subjects, block_features), noise_distribution)
        values = shared @ loadings + noise
        if noise_distribution == "heteroscedastic":
            values *= np.linspace(0.55, 1.45, n_subjects)[:, None]
        if noise_distribution == "clr":
            values -= values.mean(axis=1, keepdims=True)
        names = [f"{block_name}_f{index:04d}" for index in range(block_features)]
        signal_indices = np.arange(min(n_signal, block_features))
        stable_here = block_index in active and scenario in {
            "stable_sparse",
            "partial_views",
            "stable_plus_outlier",
            "stable_plus_contamination",
        }
        if stable_here:
            values[:, signal_indices] += effect
        if scenario == "stable_plus_outlier":
            nuisance = np.arange(n_signal, min(2 * n_signal, block_features))
            if block_index in active:
                values[0, nuisance] += effect * n_subjects
        elif scenario == "outlier":
            if block_index in active:
                values[0, signal_indices] += effect * n_subjects
        elif scenario == "heterogeneous":
            if block_index in active:
                directions = np.ones(n_subjects)
                directions[: n_subjects // 2] = -1.0
                rng.shuffle(directions)
                values[:, signal_indices] += effect * directions[:, None]
        elif scenario in {"stable_plus_contamination", "contamination_only"}:
            count = n_contamination if n_contamination is not None else n_signal
            nuisance = np.arange(n_signal, min(n_signal + count, block_features))
            amplitude = contamination_amplitude if contamination_amplitude is not None else effect * n_subjects
            values[0, nuisance] += amplitude
        elif scenario not in {"null", "stable_sparse", "partial_views", "stable_plus_outlier"}:
            raise ValueError(scenario)
        blocks[block_name] = pd.DataFrame(values, columns=names)
        truth[block_name] = (
            {names[index] for index in signal_indices} if stable_here else set()
        )
        nuisance_indices = np.arange(
            n_signal,
            min(n_signal + (n_contamination if n_contamination is not None else n_signal), block_features),
        )
        contaminants[block_name] = (
            {names[index] for index in nuisance_indices}
            if scenario in {"stable_plus_contamination", "contamination_only", "stable_plus_outlier"}
            else set()
        )
    return blocks, truth, contaminants


def _draw_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    distribution: str,
) -> np.ndarray:
    if distribution in {"normal", "heteroscedastic", "clr"}:
        return rng.normal(size=shape)
    if distribution == "t3":
        return rng.standard_t(df=3, size=shape) / np.sqrt(3.0)
    if distribution == "zero_inflated":
        values = rng.normal(size=shape)
        values[rng.random(size=shape) < 0.45] = 0.0
        return values
    if distribution == "skewed":
        values = rng.lognormal(mean=0.0, sigma=0.7, size=shape)
        return (values - np.exp(0.7**2 / 2.0)) / np.sqrt(
            (np.exp(0.7**2) - 1.0) * np.exp(0.7**2)
        )
    raise ValueError(f"Unknown noise distribution: {distribution}")


def recovery_metrics(selected: set[str], truth: set[str]) -> tuple[float, float]:
    if not truth:
        return float("nan"), float("nan")
    true_positive = len(selected & truth)
    precision = true_positive / max(len(selected), 1)
    recall = true_positive / len(truth)
    return float(precision), float(recall)
