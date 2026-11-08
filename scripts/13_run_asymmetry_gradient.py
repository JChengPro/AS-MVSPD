#!/usr/bin/env python3
"""Evaluate AS-MVSPD and published-statistic baselines across skew gradients."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import generate_signs  # noqa: E402
from stablejoint.benchmark import (  # noqa: E402
    evaluate_published_baselines,
    fast_maxT_pvalues,
    fast_stable_multiview_pvalue,
)


OUTPUT = PROJECT / "results" / "method_validation"
ALPHA = 0.05


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
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
        if "median_feature_skewness" in part:
            row["mean_median_feature_skewness"] = float(
                part["median_feature_skewness"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def tukey_g_skewness(g_value: float) -> float:
    if g_value == 0:
        return 0.0
    variance_factor = np.exp(g_value * g_value)
    return float((variance_factor + 2.0) * np.sqrt(variance_factor - 1.0))


def standardized_tukey_g(
    rng: np.random.Generator,
    shape: tuple[int, int],
    g_value: float,
) -> np.ndarray:
    normal = rng.normal(size=shape)
    if g_value == 0:
        return normal
    transformed = np.expm1(g_value * normal) / g_value
    mean = np.expm1(0.5 * g_value * g_value) / g_value
    variance = (
        np.exp(2.0 * g_value * g_value) - np.exp(g_value * g_value)
    ) / (g_value * g_value)
    return (transformed - mean) / np.sqrt(variance)


def simulate_asymmetric_multiview(
    n_subjects: int,
    n_features: int,
    n_signal: int,
    scenario: str,
    g_value: float,
    rng: np.random.Generator,
    effect: float,
    cross_view_correlation: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    if scenario not in {"null", "partial_views", "stable_sparse"}:
        raise ValueError(scenario)
    active = {0, 1} if scenario == "partial_views" else (
        {0, 1, 2} if scenario == "stable_sparse" else set()
    )
    shared = standardized_tukey_g(rng, (n_subjects, n_features), g_value)
    shared_scale = np.sqrt(cross_view_correlation)
    private_scale = np.sqrt(1.0 - cross_view_correlation)
    blocks: dict[str, pd.DataFrame] = {}
    truth: dict[str, set[str]] = {}
    directions = (1.0, -1.0, 1.0)
    for view_index in range(3):
        private = standardized_tukey_g(rng, (n_subjects, n_features), g_value)
        values = shared_scale * shared + private_scale * private
        names = [f"View{view_index + 1}_f{j:04d}" for j in range(n_features)]
        if view_index in active and n_signal:
            values[:, :n_signal] += effect * directions[view_index]
            truth[f"View{view_index + 1}"] = set(names[:n_signal])
        else:
            truth[f"View{view_index + 1}"] = set()
        blocks[f"View{view_index + 1}"] = pd.DataFrame(values, columns=names)
    return blocks, truth


def method_pvalues(blocks: dict[str, pd.DataFrame], signs: np.ndarray) -> dict[str, float]:
    return {
        "AS-MVSPD": fast_stable_multiview_pvalue(blocks, signs),
        **evaluate_published_baselines(blocks, signs),
    }


def run_joint_gradient(
    config: dict,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = config["method_validation"]
    g_values = [float(value) for value in validation["asymmetry_g_values"]]
    sample_sizes = [int(value) for value in validation["asymmetry_sample_sizes"]]
    n_features = int(validation["asymmetry_features_per_view"])
    effect = float(validation["asymmetry_effect"])
    cross_view = float(validation["asymmetry_cross_view_correlation"])
    sign_limit = int(validation["asymmetry_n12_sign_patterns"])
    signs_by_n = {}
    exact_by_n = {}
    for n_subjects in sample_sizes:
        max_random = 4095 if n_subjects == 8 else sign_limit - 1
        signs_by_n[n_subjects], exact_by_n[n_subjects] = generate_signs(
            n_subjects, max_random=max_random, seed=seed + n_subjects
        )

    rows = []
    setting = 0
    total = len(sample_sizes) * len(g_values) * 3
    for n_subjects in sample_sizes:
        signs = signs_by_n[n_subjects]
        for g_value in g_values:
            for scenario in ("null", "partial_views", "stable_sparse"):
                setting += 1
                rng = np.random.default_rng(seed + setting * 100_000)
                for replicate in range(repetitions):
                    blocks, _ = simulate_asymmetric_multiview(
                        n_subjects=n_subjects,
                        n_features=n_features,
                        n_signal=5,
                        scenario=scenario,
                        g_value=g_value,
                        rng=rng,
                        effect=effect,
                        cross_view_correlation=cross_view,
                    )
                    feature_skew = np.concatenate(
                        [
                            skew(block.to_numpy(), axis=0, bias=False, nan_policy="omit")
                            for block in blocks.values()
                        ]
                    )
                    median_skew = float(np.nanmedian(feature_skew))
                    for method, pvalue in method_pvalues(blocks, signs).items():
                        rows.append(
                            {
                                "n_subjects": n_subjects,
                                "g": g_value,
                                "theoretical_input_skewness": tukey_g_skewness(g_value),
                                "scenario": scenario,
                                "replicate": replicate,
                                "method": method,
                                "pvalue": pvalue,
                                "rejected_0_05": pvalue <= ALPHA,
                                "median_feature_skewness": median_skew,
                                "sign_patterns": len(signs),
                                "full_sign_group": exact_by_n[n_subjects],
                            }
                        )
                print(
                    f"joint asymmetry {setting}/{total}: n={n_subjects}, "
                    f"g={g_value}, {scenario}",
                    flush=True,
                )
    raw = pd.DataFrame(rows)
    summary = summarize_binary(
        raw, ["n_subjects", "g", "theoretical_input_skewness", "scenario", "method"],
        "rejected_0_05",
    )
    return raw, summary


def run_fwer_gradient(
    config: dict,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = config["method_validation"]
    g_values = [float(value) for value in validation["asymmetry_g_values"]]
    n_features = int(validation["asymmetry_fwer_features"])
    signal_counts = [int(value) for value in validation["asymmetry_signal_features"]]
    effect = float(validation["asymmetry_effect"])
    signs, exact = generate_signs(8)
    rows = []
    setting = 0
    total = len(g_values) * len(signal_counts)
    for g_value in g_values:
        for n_signal in signal_counts:
            setting += 1
            rng = np.random.default_rng(seed + setting * 100_000)
            for replicate in range(repetitions):
                values = standardized_tukey_g(rng, (8, n_features), g_value)
                if n_signal:
                    values[:, :n_signal] += effect
                columns = [f"f{j:04d}" for j in range(n_features)]
                block = pd.DataFrame(values, columns=columns)
                adjusted = fast_maxT_pvalues(block, signs)
                null_adjusted = adjusted[n_signal:]
                rows.append(
                    {
                        "n_subjects": 8,
                        "g": g_value,
                        "theoretical_input_skewness": tukey_g_skewness(g_value),
                        "n_signal": n_signal,
                        "replicate": replicate,
                        "any_false_maxT": bool((null_adjusted < ALPHA).any()),
                        "false_maxT_n": int((null_adjusted < ALPHA).sum()),
                        "median_feature_skewness": float(
                            np.nanmedian(skew(values, axis=0, bias=False, nan_policy="omit"))
                        ),
                        "sign_patterns": len(signs),
                        "full_sign_group": exact,
                    }
                )
            print(
                f"maxT asymmetry {setting}/{total}: g={g_value}, signals={n_signal}",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    summary = summarize_binary(
        raw,
        ["n_subjects", "g", "theoretical_input_skewness", "n_signal"],
        "any_false_maxT",
    )
    counts = (
        raw.groupby(
            ["n_subjects", "g", "theoretical_input_skewness", "n_signal"],
            as_index=False,
        )
        .agg(mean_false_features=("false_maxT_n", "mean"))
    )
    return raw, summary.merge(
        counts, on=["n_subjects", "g", "theoretical_input_skewness", "n_signal"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run five replicates per setting")
    args = parser.parse_args()
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    repetitions = 5 if args.quick else int(
        config["method_validation"]["asymmetry_replicates"]
    )
    seed = int(config["seed"]) + 13000
    started = time.time()
    joint_raw, joint_summary = run_joint_gradient(config, repetitions, seed + 100)
    fwer_raw, fwer_summary = run_fwer_gradient(config, repetitions, seed + 200)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joint_raw.to_csv(OUTPUT / "asymmetry_joint_replicates.csv", index=False, encoding="utf-8-sig")
    joint_summary.to_csv(OUTPUT / "asymmetry_joint_summary.csv", index=False, encoding="utf-8-sig")
    fwer_raw.to_csv(OUTPUT / "asymmetry_fwer_replicates.csv", index=False, encoding="utf-8-sig")
    fwer_summary.to_csv(OUTPUT / "asymmetry_fwer_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "status": "complete",
        "quick": bool(args.quick),
        "seed": seed,
        "replicates_per_setting": repetitions,
        "generator": "standardized Tukey-g transform of Gaussian noise",
        "g_values": config["method_validation"]["asymmetry_g_values"],
        "sample_sizes": config["method_validation"]["asymmetry_sample_sizes"],
        "joint_scenarios": ["null", "partial_views", "stable_sparse"],
        "fwer_signal_counts": config["method_validation"]["asymmetry_signal_features"],
        "methods": sorted(joint_raw["method"].unique().tolist()),
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT / "asymmetry_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
