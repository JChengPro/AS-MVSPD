#!/usr/bin/env python3
"""Validate AS-MVSPD under explicit AR(1) and block feature correlation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import generate_signs  # noqa: E402
from stablejoint.benchmark import (  # noqa: E402
    evaluate_published_baselines,
    fast_maxT_pvalues,
    fast_stable_multiview_pvalue,
    simulate_structured_correlated_multiview,
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
        total = int(len(part))
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
        "AS-MVSPD": fast_stable_multiview_pvalue(blocks, signs),
        **evaluate_published_baselines(blocks, signs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run five replicates per setting")
    args = parser.parse_args()
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    validation = config["method_validation"]
    repetitions = 5 if args.quick else int(validation["structured_correlation_replicates"])
    n_features = int(validation["structured_correlation_features"])
    n_signal = int(validation["structured_correlation_signal_features"])
    effect = float(validation["structured_correlation_effect"])
    rhos = [float(value) for value in validation["structured_correlation_rhos"]]
    structures = [str(value) for value in validation["structured_correlation_structures"]]
    cross_view = float(validation["structured_cross_view_correlation"])
    block_size = int(validation["structured_block_size"])
    seed = int(config["seed"]) + 11000

    signs_by_n: dict[int, np.ndarray] = {}
    exact_by_n: dict[int, bool] = {}
    for n_subjects in (8, 12):
        max_random = 4095 if n_subjects == 8 else int(
            validation["structured_n12_sign_patterns"]
        ) - 1
        signs_by_n[n_subjects], exact_by_n[n_subjects] = generate_signs(
            n_subjects,
            max_random=max_random,
            seed=seed + n_subjects,
        )

    global_rows: list[dict] = []
    fwer_rows: list[dict] = []
    started = time.time()
    setting_index = 0
    scenarios = ("null", "partial_views", "stable_sparse")
    total_settings = 2 * len(structures) * len(rhos) * len(scenarios)
    for n_subjects in (8, 12):
        signs = signs_by_n[n_subjects]
        for structure in structures:
            for rho in rhos:
                for scenario in scenarios:
                    setting_index += 1
                    rng = np.random.default_rng(seed + setting_index * 100_000)
                    for replicate in range(repetitions):
                        blocks, truth = simulate_structured_correlated_multiview(
                            n_subjects=n_subjects,
                            n_features=n_features,
                            n_signal=n_signal,
                            scenario=scenario,
                            rng=rng,
                            structure=structure,
                            rho=rho,
                            effect=effect,
                            n_views=3,
                            cross_view_correlation=cross_view,
                            block_size=block_size,
                            signal_directions=(1, -1, 1),
                            active_views=(0, 1) if scenario == "partial_views" else None,
                        )
                        for method, pvalue in method_pvalues(blocks, signs).items():
                            global_rows.append(
                                {
                                    "n_subjects": n_subjects,
                                    "structure": structure,
                                    "rho": rho,
                                    "scenario": scenario,
                                    "replicate": replicate,
                                    "method": method,
                                    "pvalue": pvalue,
                                    "rejected_0_05": pvalue <= ALPHA,
                                    "sign_patterns": len(signs),
                                    "full_sign_group": exact_by_n[n_subjects],
                                }
                            )
                        if scenario == "stable_sparse":
                            for view, block in blocks.items():
                                adjusted = fast_maxT_pvalues(block, signs)
                                null_indices = [
                                    index
                                    for index, feature in enumerate(block.columns)
                                    if feature not in truth[view]
                                ]
                                null_p = adjusted[null_indices]
                                fwer_rows.append(
                                    {
                                        "n_subjects": n_subjects,
                                        "structure": structure,
                                        "rho": rho,
                                        "replicate": replicate,
                                        "view": view,
                                        "any_false_maxT": bool((null_p < ALPHA).any()),
                                        "false_maxT_n": int((null_p < ALPHA).sum()),
                                        "sign_patterns": len(signs),
                                        "full_sign_group": exact_by_n[n_subjects],
                                    }
                                )
                    print(
                        f"completed {setting_index}/{total_settings}: "
                        f"n={n_subjects}, {structure}, rho={rho}, {scenario}",
                        flush=True,
                    )

    global_raw = pd.DataFrame(global_rows)
    fwer_raw = pd.DataFrame(fwer_rows)
    global_summary = summarize_binary(
        global_raw,
        ["n_subjects", "structure", "rho", "scenario", "method"],
        "rejected_0_05",
    )
    fwer_summary = summarize_binary(
        fwer_raw,
        ["n_subjects", "structure", "rho", "view"],
        "any_false_maxT",
    )
    false_counts = (
        fwer_raw.groupby(["n_subjects", "structure", "rho", "view"], as_index=False)
        .agg(mean_false_features=("false_maxT_n", "mean"))
    )
    fwer_summary = fwer_summary.merge(
        false_counts, on=["n_subjects", "structure", "rho", "view"]
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    global_raw.to_csv(
        OUTPUT / "structured_correlation_global_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_summary.to_csv(
        OUTPUT / "structured_correlation_global_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fwer_raw.to_csv(
        OUTPUT / "structured_correlation_fwer_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fwer_summary.to_csv(
        OUTPUT / "structured_correlation_fwer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "status": "complete",
        "quick": bool(args.quick),
        "seed": seed,
        "replicates_per_setting": repetitions,
        "sample_sizes": [8, 12],
        "n_views": 3,
        "n_features_per_view": n_features,
        "n_signal_per_view": n_signal,
        "effect": effect,
        "signal_directions": [1, -1, 1],
        "partial_null_active_views": [1, 2],
        "structures": structures,
        "rhos": rhos,
        "cross_view_correlation": cross_view,
        "block_size": block_size,
        "sign_patterns": {str(n): len(signs_by_n[n]) for n in signs_by_n},
        "full_sign_group": {str(n): exact_by_n[n] for n in exact_by_n},
        "elapsed_seconds": time.time() - started,
    }
    (OUTPUT / "structured_correlation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved structured-correlation validation under {OUTPUT}")


if __name__ == "__main__":
    main()
