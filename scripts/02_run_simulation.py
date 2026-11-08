#!/usr/bin/env python3
"""Benchmark AS-MVSPD under stable, null and unstable paired shifts."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint.benchmark import evaluate_methods, recovery_metrics, simulate_paired_blocks  # noqa: E402
from stablejoint.core import generate_signs  # noqa: E402


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="Three replicates per setting, written under results/quick")
    args=parser.parse_args()
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    replicates = 3 if args.quick else int(config["simulation_replicates"])
    sample_sizes = [int(value) for value in config["simulation_sample_sizes"]]
    n_features = int(config["simulation_features_per_block"])
    n_signal = int(config["simulation_signal_features"])
    alpha = float(config["simulation_alpha"])
    k_grid = tuple(int(value) for value in config["k_grid"])
    scenarios = (
        "null",
        "stable_sparse",
        "stable_plus_outlier",
        "outlier",
        "one_block",
        "heterogeneous",
    )
    rng = np.random.default_rng(seed)
    rows = []
    recovery_rows = []

    for n_subjects in sample_sizes:
        max_random = 255 if n_subjects > 8 else int(config["max_random_sign_flips"])
        signs, exact = generate_signs(n_subjects, max_random=max_random, seed=seed + n_subjects)
        for scenario in scenarios:
            for replicate in range(replicates):
                blocks, truth = simulate_paired_blocks(
                    n_subjects=n_subjects,
                    n_features=n_features,
                    n_signal=n_signal,
                    scenario=scenario,
                    rng=rng,
                )
                pvalues, selected = evaluate_methods(blocks, signs, k_grid=k_grid)
                for method, pvalue in pvalues.items():
                    rows.append(
                        {
                            "n_subjects": n_subjects,
                            "scenario": scenario,
                            "replicate": replicate,
                            "method": method,
                            "pvalue": pvalue,
                            "rejected_0_05": pvalue <= alpha,
                            "sign_patterns": len(signs),
                            "exact_enumeration": exact,
                        }
                    )
                if scenario in {"stable_sparse", "stable_plus_outlier"}:
                    for selection_method, method_selections in selected.items():
                        for block_name in blocks:
                            precision, recall = recovery_metrics(
                                method_selections[block_name], truth[block_name]
                            )
                            recovery_rows.append(
                                {
                                    "n_subjects": n_subjects,
                                    "scenario": scenario,
                                    "replicate": replicate,
                                    "selection_method": selection_method,
                                    "block": block_name,
                                    "selected_n": len(method_selections[block_name]),
                                    "precision": precision,
                                    "recall": recall,
                                }
                            )

    output = PROJECT / "results" / ("quick/simulation" if args.quick else "simulation")
    output.mkdir(parents=True, exist_ok=True)
    replicate_results = pd.DataFrame(rows)
    replicate_results.to_csv(output / "replicate_results.csv", index=False, encoding="utf-8-sig")
    summary = (
        replicate_results.groupby(["n_subjects", "scenario", "method"], as_index=False)
        .agg(
            rejection_rate=("rejected_0_05", "mean"),
            median_p=("pvalue", "median"),
            replicates=("replicate", "nunique"),
        )
    )
    summary.to_csv(output / "performance_summary.csv", index=False, encoding="utf-8-sig")
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(output / "feature_recovery.csv", index=False, encoding="utf-8-sig")
    recovery_summary = (
        recovery.groupby(
            ["n_subjects", "scenario", "selection_method", "block"], as_index=False
        )
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            median_selected_n=("selected_n", "median"),
        )
    )
    recovery_summary.to_csv(
        output / "feature_recovery_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))
    print(recovery_summary.to_string(index=False))


if __name__ == "__main__":
    main()
