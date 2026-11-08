import numpy as np
import pandas as pd

from stablejoint.benchmark import (
    _paired_aspu_adaptive_evidence,
    _paired_aspu_block_pvalue,
    _paired_cq_block_pvalue,
    _paired_cq_statistics,
    _paired_rankcom_adaptive_evidence,
    _paired_rankcom_block_pvalue,
    _signed_rank_scores,
    evaluate_published_baselines,
    fast_maxT_pvalues,
    fast_stable_multiview_pvalue,
    simulate_structured_correlated_multiview,
    structured_correlation_factor,
)
from stablejoint.core import (
    _tail_probabilities,
    analyze_multiview,
    generate_signs,
    self_normalized_feature_scores,
)


def test_outlier_is_suppressed_by_influence_constraint():
    delta = np.array([[12.0], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
    signs, exact = generate_signs(8)
    unconstrained = self_normalized_feature_scores(delta, signs, False)[0, 0]
    constrained = self_normalized_feature_scores(delta, signs, True)[0, 0]
    assert exact
    assert constrained < unconstrained


def test_stable_shift_has_minimum_exact_joint_pvalue():
    index = pd.Index([f"s{i}" for i in range(8)], name="subject")
    block_a = pd.DataFrame({"a": np.arange(1.0, 9.0)}, index=index)
    block_b = pd.DataFrame({"b": np.arange(2.0, 10.0)}, index=index)
    result = analyze_multiview({"a": block_a, "b": block_b})
    assert result.joint_pvalue <= 0.05


def test_one_block_signal_does_not_pass_conjunction():
    index = pd.Index([f"s{i}" for i in range(8)], name="subject")
    block_a = pd.DataFrame({"a": np.arange(1.0, 9.0)}, index=index)
    block_b = pd.DataFrame({"b": [-1.0, 1.0] * 4}, index=index)
    result = analyze_multiview({"a": block_a, "b": block_b})
    assert result.joint_pvalue > 0.05


def test_sampled_signs_form_closed_subgroup():
    signs, exact = generate_signs(20, max_random=255, seed=17)
    assert not exact
    assert len(signs) == 256
    assert np.all(signs[0] == 1.0)
    encoded = {tuple(row.astype(int)) for row in signs}
    for left in signs[:20]:
        for right in signs[:20]:
            assert tuple((left * right).astype(int)) in encoded


def test_n12_can_be_enumerated_exactly():
    signs, exact = generate_signs(12, max_random=4095, seed=17)
    assert exact
    assert len(signs) == 4096


def test_published_baseline_pvalues_are_valid_bounds():
    index = pd.Index([f"s{i}" for i in range(8)], name="subject")
    blocks = {
        "a": pd.DataFrame({"a": np.arange(1.0, 9.0), "b": np.arange(2.0, 10.0)}, index=index),
        "b": pd.DataFrame({"c": np.arange(1.5, 9.5), "d": np.arange(3.0, 11.0)}, index=index),
    }
    signs, _ = generate_signs(8)
    result = evaluate_published_baselines(blocks, signs)
    assert set(result) == {
        "Paired-aSPU-IUT",
        "Paired-CQ-IUT",
        "Paired-RankCOM-IUT",
    }
    assert all(0.0 < pvalue <= 1.0 for pvalue in result.values())


def test_cq_orbit_statistics_match_pairwise_u_statistic():
    values = np.array(
        [[1.0, -0.5, 2.0], [0.2, 1.4, -0.3], [1.7, 0.1, 0.8], [-0.4, 0.9, 1.1]]
    )
    signs, _ = generate_signs(len(values))
    observed = _paired_cq_statistics(values, signs)
    reference = []
    for pattern in signs:
        signed = values * pattern[:, None]
        pair_sum = sum(
            float(np.dot(signed[i], signed[j]))
            for i in range(len(values))
            for j in range(len(values))
            if i != j
        )
        reference.append(pair_sum / (len(values) * (len(values) - 1)))
    np.testing.assert_allclose(observed, reference, atol=1e-12)
    assert _paired_cq_block_pvalue(values, signs) == _tail_probabilities(observed)[0]


def test_aspu_single_power_matches_manual_orbit_calibration():
    values = np.array(
        [[0.5, -1.0], [1.2, 0.4], [-0.3, 1.5], [0.8, 0.6]]
    )
    signs, _ = generate_signs(len(values))
    denominator = np.sqrt(np.sum(values * values, axis=0))
    scores = (signs @ values) / denominator
    statistic = np.sum(scores**2, axis=1)
    scale_p = _tail_probabilities(statistic)
    reference = -np.log(np.maximum(scale_p, 1.0 / len(signs)))
    observed = _paired_aspu_adaptive_evidence(values, signs, powers=(2,))
    np.testing.assert_allclose(observed, reference, atol=1e-12)
    assert _paired_aspu_block_pvalue(values, signs, powers=(2,)) == _tail_probabilities(reference)[0]


def test_published_baseline_iut_equals_largest_component_pvalue():
    index = pd.Index([f"s{i}" for i in range(8)], name="subject")
    block_a = pd.DataFrame({"a": np.arange(1.0, 9.0), "b": np.arange(2.0, 10.0)}, index=index)
    block_b = pd.DataFrame({"c": [-1.0, 1.0] * 4, "d": [1.0, -1.0] * 4}, index=index)
    signs, _ = generate_signs(8)
    combined = evaluate_published_baselines({"a": block_a, "b": block_b}, signs)
    assert combined["Paired-CQ-IUT"] == max(
        _paired_cq_block_pvalue(block_a.to_numpy(), signs),
        _paired_cq_block_pvalue(block_b.to_numpy(), signs),
    )
    assert combined["Paired-aSPU-IUT"] == max(
        _paired_aspu_block_pvalue(block_a.to_numpy(), signs),
        _paired_aspu_block_pvalue(block_b.to_numpy(), signs),
    )
    assert combined["Paired-RankCOM-IUT"] == max(
        _paired_rankcom_block_pvalue(block_a.to_numpy(), signs),
        _paired_rankcom_block_pvalue(block_b.to_numpy(), signs),
    )


def test_rankcom_signed_ranks_and_orbit_calibration_match_manual_calculation():
    values = np.array(
        [[1.0, -4.0], [3.0, 0.0], [-2.0, 2.0], [4.0, -1.0]]
    )
    signs, _ = generate_signs(len(values))
    expected_signed_ranks = np.array(
        [[1.0, -3.0], [3.0, 0.0], [-2.0, 2.0], [4.0, -1.0]]
    )
    np.testing.assert_allclose(_signed_rank_scores(values), expected_signed_ranks)

    denominator = np.sqrt(np.sum(expected_signed_ranks**2, axis=0))
    standardized = (signs @ expected_signed_ranks) / denominator
    component_p = np.column_stack(
        [
            _tail_probabilities(np.max(np.abs(standardized), axis=1)),
            _tail_probabilities(np.mean(standardized**2, axis=1)),
        ]
    )
    clipped = np.clip(component_p, 0.5 / len(signs), 1.0 - 0.5 / len(signs))
    cauchy = np.mean(np.tan((0.5 - clipped) * np.pi), axis=1)
    combined_p = 0.5 - np.arctan(cauchy) / np.pi
    reference = -np.log(np.maximum(combined_p, 1.0 / len(signs)))
    observed = _paired_rankcom_adaptive_evidence(values, signs)
    np.testing.assert_allclose(observed, reference, atol=1e-12)
    assert _paired_rankcom_block_pvalue(values, signs) == _tail_probabilities(reference)[0]


def test_tail_probabilities_retain_ties_conservatively():
    values = np.array([3.0, 3.0, 2.0, 1.0])
    observed = _tail_probabilities(values)
    np.testing.assert_allclose(observed, [0.5, 0.5, 0.75, 1.0])


def test_fast_paths_match_full_analysis():
    rng = np.random.default_rng(31)
    index = pd.Index([f"s{i}" for i in range(8)], name="subject")
    blocks = {
        "a": pd.DataFrame(rng.normal(size=(8, 12)), index=index),
        "b": pd.DataFrame(rng.normal(size=(8, 17)), index=index),
    }
    signs, _ = generate_signs(8)
    for constrained in (True, False):
        full = analyze_multiview(
            blocks, signs=signs, influence_constraint=constrained
        )
        fast = fast_stable_multiview_pvalue(
            blocks, signs, influence_constraint=constrained
        )
        assert fast == full.joint_pvalue
        expected = full.block_results["a"].feature_table.set_index("feature").loc[
            blocks["a"].columns.astype(str), "maxT_fwer"
        ]
        observed = fast_maxT_pvalues(
            blocks["a"], signs, influence_constraint=constrained
        )
        np.testing.assert_allclose(observed, expected.to_numpy())


def test_structured_correlation_factors_match_requested_covariance():
    ar1 = structured_correlation_factor(8, "ar1", 0.7)
    ar1_covariance = ar1 @ ar1.T
    assert np.allclose(np.diag(ar1_covariance), 1.0)
    assert np.allclose(ar1_covariance[0, :4], [1.0, 0.7, 0.49, 0.343])

    block = structured_correlation_factor(9, "block", 0.3, block_size=4)
    block_covariance = block @ block.T
    assert np.allclose(np.diag(block_covariance), 1.0)
    assert np.isclose(block_covariance[0, 3], 0.3)
    assert np.isclose(block_covariance[0, 4], 0.0)


def test_joint_pvalue_is_invariant_to_reversing_one_view_direction():
    rng = np.random.default_rng(1024)
    blocks, _ = simulate_structured_correlated_multiview(
        n_subjects=8,
        n_features=30,
        n_signal=5,
        scenario="stable_sparse",
        rng=rng,
        structure="ar1",
        rho=0.7,
        signal_directions=(1, -1, 1),
    )
    signs, _ = generate_signs(8)
    original = analyze_multiview(blocks, signs=signs)
    reversed_blocks = dict(blocks)
    reversed_blocks["View2"] = -reversed_blocks["View2"]
    reversed_result = analyze_multiview(reversed_blocks, signs=signs)
    assert reversed_result.joint_pvalue == original.joint_pvalue
    assert np.allclose(
        reversed_result.block_results["View2"].feature_table["stable_score"],
        original.block_results["View2"].feature_table["stable_score"],
    )
