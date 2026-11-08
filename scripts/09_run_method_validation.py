#!/usr/bin/env python3
"""Run targeted validation required for the AS-MVSPD methods manuscript."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import analyze_block, analyze_multiview, generate_signs  # noqa: E402
from stablejoint.benchmark import (  # noqa: E402
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
    simulate_paired_multiview,
)
from stablejoint.core import _tail_probabilities  # noqa: E402


OUTPUT = PROJECT / "results" / "method_validation"
ALPHA = 0.05


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total**2))
    half /= denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_binary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, sort=False, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        successes = int(part[value].astype(bool).sum())
        total = len(part)
        low, high = wilson_interval(successes, total)
        row = dict(zip(groups, key_tuple))
        row.update(
            successes=successes,
            replicates=total,
            rate=successes / total,
            ci_low=low,
            ci_high=high,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def method_pvalues(blocks: dict[str, pd.DataFrame], signs: np.ndarray) -> dict[str, float]:
    return {
        "AS-MVSPD": fast_stable_multiview_pvalue(
            blocks, signs, influence_constraint=False
        ),
        "AS-MVSPD+LOO": fast_stable_multiview_pvalue(
            blocks, signs, influence_constraint=True
        ),
        **evaluate_published_baselines(blocks, signs),
    }


def run_three_view_iut(replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    signs, exact = generate_signs(8)
    rows = []
    for active_count in range(4):
        scenario = "null" if active_count == 0 else "partial_views"
        active_views = tuple(range(active_count))
        for replicate in range(replicates):
            blocks, _, _ = simulate_paired_multiview(
                n_subjects=8,
                n_features=(100, 200, 400),
                n_signal=5,
                scenario=scenario,
                rng=rng,
                effect=1.8,
                n_views=3,
                active_views=active_views,
            )
            for method, pvalue in method_pvalues(blocks, signs).items():
                rows.append(
                    {
                        "active_views": active_count,
                        "scenario": f"{active_count}/3 views",
                        "replicate": replicate,
                        "method": method,
                        "pvalue": pvalue,
                        "rejected_0_05": pvalue <= ALPHA,
                        "sign_patterns": len(signs),
                        "full_group": exact,
                    }
                )
    raw = pd.DataFrame(rows)
    summary = summarize_binary(raw, ["active_views", "scenario", "method"], "rejected_0_05")
    return raw, summary


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def run_influence_validation(replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    signs, _ = generate_signs(8)
    loo_signs, _ = generate_signs(7)
    rows = []
    for scenario in ("contamination_only", "stable_plus_contamination"):
        for amplitude in (4.0, 8.0, 12.0):
            for replicate in range(replicates):
                blocks, truth, contaminants = simulate_paired_multiview(
                    n_subjects=8,
                    n_features=200,
                    n_signal=5,
                    scenario=scenario,
                    rng=rng,
                    n_views=3,
                    contamination_amplitude=amplitude,
                    n_contamination=5,
                )
                for method, constrained in (("AS-MVSPD", False), ("AS-MVSPD+LOO", True)):
                    full = analyze_multiview(
                        blocks, signs=signs, influence_constraint=constrained
                    )
                    loo_blocks = {name: block.iloc[1:].copy() for name, block in blocks.items()}
                    loo = analyze_multiview(
                        loo_blocks, signs=loo_signs, influence_constraint=constrained
                    )
                    for view_name in blocks:
                        full_block = full.block_results[view_name]
                        loo_block = loo.block_results[view_name]
                        selected = set(full_block.selected_features)
                        stable_truth = truth[view_name]
                        nuisance = contaminants[view_name]
                        full_table = full_block.feature_table.set_index("feature")
                        loo_table = loo_block.feature_table.set_index("feature")
                        top_full = set(full_table.nsmallest(10, "stable_rank").index)
                        top_loo = set(loo_table.nsmallest(10, "stable_rank").index)
                        rank_corr = float(
                            spearmanr(
                                full_table.loc[loo_table.index, "stable_rank"],
                                loo_table["stable_rank"],
                            ).statistic
                        )
                        nuisance_fwer = bool(
                            (full_table.loc[list(nuisance), "maxT_fwer"] < ALPHA).any()
                        )
                        rows.append(
                            {
                                "scenario": scenario,
                                "amplitude": amplitude,
                                "replicate": replicate,
                                "method": method,
                                "view": view_name,
                                "joint_rejected": full.joint_pvalue <= ALPHA,
                                "selected_n": len(selected),
                                "stable_precision": (
                                    len(selected & stable_truth) / max(len(selected), 1)
                                    if stable_truth
                                    else float("nan")
                                ),
                                "stable_recall": (
                                    len(selected & stable_truth) / len(stable_truth)
                                    if stable_truth
                                    else float("nan")
                                ),
                                "contaminant_selected_fraction": len(selected & nuisance)
                                / len(nuisance),
                                "any_contaminant_maxT": nuisance_fwer,
                                "top10_jaccard_after_loo": _jaccard(top_full, top_loo),
                                "rank_spearman_after_loo": rank_corr,
                            }
                        )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["scenario", "amplitude", "method"], as_index=False)
        .agg(
            replicates=("replicate", "nunique"),
            joint_rejection_rate=("joint_rejected", "mean"),
            stable_precision=("stable_precision", "mean"),
            stable_recall=("stable_recall", "mean"),
            contaminant_selected_fraction=("contaminant_selected_fraction", "mean"),
            contaminant_maxT_rate=("any_contaminant_maxT", "mean"),
            top10_jaccard_after_loo=("top10_jaccard_after_loo", "mean"),
            rank_spearman_after_loo=("rank_spearman_after_loo", "mean"),
        )
    )
    return raw, summary


def run_partial_null_fwer(replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    signs, _ = generate_signs(8)
    rows = []
    for distribution in (
        "normal",
        "t3",
        "zero_inflated",
        "heteroscedastic",
        "clr",
        "skewed",
    ):
        for n_signal in (0, 5, 20):
            for replicate in range(replicates):
                blocks, truth, _ = simulate_paired_multiview(
                    n_subjects=8,
                    n_features=200,
                    n_signal=n_signal,
                    scenario="stable_sparse",
                    rng=rng,
                    n_views=1,
                    noise_distribution=distribution,
                )
                block = blocks["View1"]
                null_features = [x for x in block.columns if x not in truth["View1"]]
                null_indices = block.columns.get_indexer(null_features)
                for method, constrained in (("AS-MVSPD", False), ("AS-MVSPD+LOO", True)):
                    adjusted = fast_maxT_pvalues(
                        block, signs=signs, influence_constraint=constrained
                    )
                    null_p = adjusted[null_indices]
                    rows.append(
                        {
                            "distribution": distribution,
                            "n_signal": n_signal,
                            "replicate": replicate,
                            "method": method,
                            "any_false_maxT": bool((null_p < ALPHA).any()),
                            "false_maxT_n": int((null_p < ALPHA).sum()),
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = summarize_binary(
        raw, ["distribution", "n_signal", "method"], "any_false_maxT"
    )
    counts = (
        raw.groupby(["distribution", "n_signal", "method"], as_index=False)
        .agg(mean_false_features=("false_maxT_n", "mean"))
    )
    return raw, summary.merge(counts, on=["distribution", "n_signal", "method"])


def run_published_baseline_benchmark(
    replicates: int, seed: int, n12_sign_patterns: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for n_subjects in (8, 12):
        max_random = 4095 if n_subjects == 12 and n12_sign_patterns >= 4096 else 255
        signs, exact = generate_signs(
            n_subjects, max_random=max_random, seed=seed + n_subjects
        )
        for n_signal in (1, 5, 20):
            for replicate in range(replicates):
                blocks, _, _ = simulate_paired_multiview(
                    n_subjects=n_subjects,
                    n_features=100,
                    n_signal=n_signal,
                    scenario="stable_sparse",
                    rng=rng,
                    n_views=3,
                )
                for method, pvalue in method_pvalues(blocks, signs).items():
                    rows.append(
                        {
                            "n_subjects": n_subjects,
                            "n_signal": n_signal,
                            "replicate": replicate,
                            "method": method,
                            "pvalue": pvalue,
                            "rejected_0_05": pvalue <= ALPHA,
                            "sign_patterns": len(signs),
                            "full_group": exact,
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = summarize_binary(
        raw, ["n_subjects", "n_signal", "method"], "rejected_0_05"
    )
    return raw, summary


def load_public_blocks():
    from stablejoint.public_datasets import iter_public_subsets
    return [(label,{'Microbiome':x,'Metabolome':y}) for _,_,label,x,y in iter_public_subsets(PROJECT/'data/public_prepared')]



def run_real_data_baselines(seed: int) -> pd.DataFrame:
    rows = []
    for label, blocks in load_public_blocks():
        n_subjects = len(next(iter(blocks.values())))
        signs, exact = generate_signs(
            n_subjects,
            max_random=255 if n_subjects > 8 else 4095,
            seed=seed + n_subjects,
        )
        for method, pvalue in method_pvalues(blocks, signs).items():
            rows.append(
                {
                    "data_scope": "public",
                    "dataset": label,
                    "n_subjects": n_subjects,
                    "method": method,
                    "joint_p": pvalue,
                    "sign_patterns": len(signs),
                    "full_group": exact,
                }
            )

    local_root = PROJECT / "data" / "local"
    if not (local_root/"species_paired_changes.csv").exists():
        print("Rat comparison not run: local matrices not supplied", flush=True)
        return pd.DataFrame(rows)
    local_blocks = {
        "Species": pd.read_csv(local_root / "species_paired_changes.csv", index_col=0),
        "KO": pd.read_csv(local_root / "ko_paired_changes.csv", index_col=0),
        "Metabolite": pd.read_csv(local_root / "metabolite_paired_changes.csv", index_col=0),
    }
    first_index = next(iter(local_blocks.values())).index
    for block in local_blocks.values():
        block.index = block.index.astype(str)
    first_index = next(iter(local_blocks.values())).index
    if any(not first_index.equals(block.index) for block in local_blocks.values()):
        raise ValueError("Local multi-view subject order is not aligned")
    signs, exact = generate_signs(len(first_index))
    for method, pvalue in method_pvalues(local_blocks, signs).items():
        rows.append(
            {
                "data_scope": "local",
                "dataset": "Local high-pressure rats",
                "n_subjects": len(first_index),
                "method": method,
                "joint_p": pvalue,
                "sign_patterns": len(signs),
                "full_group": exact,
            }
        )
    return pd.DataFrame(rows)


def save_pair(raw: pd.DataFrame, summary: pd.DataFrame, stem: str) -> None:
    raw.to_csv(OUTPUT / f"{stem}_replicates.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / f"{stem}_summary.csv", index=False, encoding="utf-8-sig")


def run_baseline_formula_checks() -> pd.DataFrame:
    """Cross-check baseline formulas against independent direct calculations."""

    values = np.array(
        [[1.0, -0.5, 2.0], [0.2, 1.4, -0.3], [1.7, 0.1, 0.8], [-0.4, 0.9, 1.1]]
    )
    signs, _ = generate_signs(len(values))

    cq_observed = _paired_cq_statistics(values, signs)
    cq_reference = []
    for pattern in signs:
        signed = values * pattern[:, None]
        pair_sum = sum(
            float(np.dot(signed[i], signed[j]))
            for i in range(len(values))
            for j in range(len(values))
            if i != j
        )
        cq_reference.append(pair_sum / (len(values) * (len(values) - 1)))
    cq_reference = np.asarray(cq_reference)
    cq_p_reference = float(_tail_probabilities(cq_reference)[0])
    cq_p_observed = _paired_cq_block_pvalue(values, signs)

    denominator = np.sqrt(np.sum(values * values, axis=0))
    standardized = (signs @ values) / denominator
    power2_statistic = np.sum(standardized**2, axis=1)
    power2_tail = _tail_probabilities(power2_statistic)
    aspu_reference = -np.log(np.maximum(power2_tail, 1.0 / len(signs)))
    aspu_observed = _paired_aspu_adaptive_evidence(values, signs, powers=(2,))
    aspu_p_reference = float(_tail_probabilities(aspu_reference)[0])
    aspu_p_observed = _paired_aspu_block_pvalue(values, signs, powers=(2,))

    signed_ranks = _signed_rank_scores(values)
    rank_denominator = np.sqrt(np.sum(signed_ranks * signed_ranks, axis=0))
    rank_standardized = (signs @ signed_ranks) / rank_denominator
    rank_component_p = np.column_stack(
        [
            _tail_probabilities(np.max(np.abs(rank_standardized), axis=1)),
            _tail_probabilities(np.mean(rank_standardized**2, axis=1)),
        ]
    )
    rank_clipped = np.clip(
        rank_component_p, 0.5 / len(signs), 1.0 - 0.5 / len(signs)
    )
    rank_cauchy = np.mean(np.tan((0.5 - rank_clipped) * np.pi), axis=1)
    rank_combined_p = 0.5 - np.arctan(rank_cauchy) / np.pi
    rank_reference = -np.log(np.maximum(rank_combined_p, 1.0 / len(signs)))
    rank_observed = _paired_rankcom_adaptive_evidence(values, signs)
    rank_p_reference = float(_tail_probabilities(rank_reference)[0])
    rank_p_observed = _paired_rankcom_block_pvalue(values, signs)

    second = values[:, ::-1] * np.array([1.0, 0.7, 1.2])
    blocks = {
        "view1": pd.DataFrame(values),
        "view2": pd.DataFrame(second),
    }
    combined = evaluate_published_baselines(blocks, signs)
    cq_iut_reference = max(
        _paired_cq_block_pvalue(values, signs),
        _paired_cq_block_pvalue(second, signs),
    )
    aspu_iut_reference = max(
        _paired_aspu_block_pvalue(values, signs),
        _paired_aspu_block_pvalue(second, signs),
    )
    rank_iut_reference = max(
        _paired_rankcom_block_pvalue(values, signs),
        _paired_rankcom_block_pvalue(second, signs),
    )

    checks = [
        (
            "chen_qin_orbit_statistics",
            float(np.max(np.abs(cq_observed - cq_reference))),
            0.0,
            "closed-form orbit statistics versus direct ordered-pair U-statistic",
        ),
        (
            "chen_qin_orbit_pvalue",
            cq_p_observed,
            cq_p_reference,
            "implementation P versus direct orbit-tail P",
        ),
        (
            "aspu_power2_evidence",
            float(np.max(np.abs(aspu_observed - aspu_reference))),
            0.0,
            "single-power adaptive evidence versus direct standardized sum of squares",
        ),
        (
            "aspu_power2_orbit_pvalue",
            aspu_p_observed,
            aspu_p_reference,
            "implementation P versus direct nested orbit calibration",
        ),
        (
            "chen_qin_iut_maximum",
            combined["Paired-CQ-IUT"],
            cq_iut_reference,
            "multi-view P versus maximum component P",
        ),
        (
            "aspu_iut_maximum",
            combined["Paired-aSPU-IUT"],
            aspu_iut_reference,
            "multi-view P versus maximum component P",
        ),
        (
            "rankcom_orbit_evidence",
            float(np.max(np.abs(rank_observed - rank_reference))),
            0.0,
            "signed-rank max/sum Cauchy evidence versus direct calculation",
        ),
        (
            "rankcom_orbit_pvalue",
            rank_p_observed,
            rank_p_reference,
            "rank-combination implementation P versus direct orbit-tail P",
        ),
        (
            "rankcom_iut_maximum",
            combined["Paired-RankCOM-IUT"],
            rank_iut_reference,
            "multi-view rank-combination P versus maximum component P",
        ),
    ]
    rows = []
    for name, observed, reference, description in checks:
        error = abs(observed - reference)
        rows.append(
            {
                "check": name,
                "implementation_value": observed,
                "reference_value": reference,
                "absolute_error": error,
                "tolerance": 1e-12,
                "passed": error <= 1e-12,
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test design")
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Run deterministic baseline formula checks without rerunning simulations",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Rerun only three-view and published-baseline simulations",
    )
    parser.add_argument("--simulations-only",action="store_true",help="Skip data-dependent public and rat comparisons")
    args = parser.parse_args()
    global OUTPUT
    if args.quick: OUTPUT = PROJECT / "results/quick/method_validation"
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    validation = config["method_validation"]
    seed = int(config["seed"]) + 9000
    if args.quick:
        counts = {"three": 5, "influence": 3, "fwer": 5, "baseline": 3}
    else:
        counts = {
            "three": int(validation["three_view_replicates"]),
            "influence": int(validation["influence_replicates"]),
            "fwer": int(validation["partial_fwer_replicates"]),
            "baseline": int(validation["published_baseline_replicates"]),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    formula_checks = run_baseline_formula_checks()
    formula_checks.to_csv(
        OUTPUT / "baseline_formula_checks.csv", index=False, encoding="utf-8-sig"
    )
    if not formula_checks["passed"].all():
        raise RuntimeError("At least one baseline formula check failed")
    print("baseline formula checks complete", flush=True)
    if args.checks_only:
        metadata_path = OUTPUT / "run_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        metadata["baseline_formula_checks"] = {
            "checks": int(len(formula_checks)),
            "all_passed": True,
            "maximum_absolute_error": float(formula_checks["absolute_error"].max()),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(formula_checks.to_string(index=False), flush=True)
        return
    started = time.time()

    raw, summary = run_three_view_iut(counts["three"], seed + 100)
    save_pair(raw, summary, "three_view_iut")
    print("three-view validation complete", flush=True)

    if not args.core_only:
        raw, summary = run_influence_validation(counts["influence"], seed + 200)
        save_pair(raw, summary, "influence_robustness")
        print("influence validation complete", flush=True)

        raw, summary = run_partial_null_fwer(counts["fwer"], seed + 300)
        save_pair(raw, summary, "partial_null_fwer")
        print("partial-null FWER validation complete", flush=True)

    raw, summary = run_published_baseline_benchmark(
        counts["baseline"], seed + 400, int(validation["n12_sign_patterns"])
    )
    save_pair(raw, summary, "published_baseline")
    print("published baseline simulation complete", flush=True)

    # Real-data comparisons use the same prespecified sign subgroup as the
    # primary public-data workflow so identical datasets have identical P values.
    real = pd.DataFrame() if args.simulations_only else run_real_data_baselines(int(config["seed"]))
    if not args.simulations_only:
        real.to_csv(OUTPUT / "real_data_published_baselines.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "seed": seed,
        "quick": args.quick,
        "core_only": args.core_only,
        "sections_run": (
            ["three_view_iut", "published_baseline", "real_data_baselines"]
            if args.core_only
            else [
                "three_view_iut",
                "influence_robustness",
                "partial_null_fwer",
                "published_baseline",
                "real_data_baselines",
            ]
        ),
        "replicates": counts,
        "elapsed_seconds": time.time() - started,
        "sign_generation": "complete full group or complete random GF(2) subgroup",
        "published_baselines": {
            "Paired-aSPU-IUT": "paired sign-flip adaptation of Xu et al. 2016 adaptive sum-of-powers",
            "Paired-CQ-IUT": "paired one-sample sign-flip adaptation of Chen and Qin 2010 quadratic statistic",
            "Paired-RankCOM-IUT": "paired closed-orbit adaptation of Zhang and Feng 2024 signed-rank max/sum Cauchy combination",
        },
        "baseline_formula_checks": {
            "checks": int(len(formula_checks)),
            "all_passed": bool(formula_checks["passed"].all()),
            "maximum_absolute_error": float(formula_checks["absolute_error"].max()),
        },
        "primary_method": "AS-MVSPD",
        "loo_variant": "AS-MVSPD+LOO retained only as an ablation because targeted validation showed no advantage",
    }
    if args.simulations_only:
        metadata["sections_run"] = [s for s in metadata["sections_run"] if s != "real_data_baselines"]
    (OUTPUT / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(real.to_string(index=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
