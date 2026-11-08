#!/usr/bin/env python3
"""Apply the locked AS-MVSPD test to local Species, KO and metabolite data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import analyze_multiview, generate_signs  # noqa: E402


def main() -> None:
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    source = PROJECT / "data" / "local"
    blocks = {
        "Species": pd.read_csv(source / "species_paired_changes.csv", index_col=0),
        "KO": pd.read_csv(source / "ko_paired_changes.csv", index_col=0),
        "Metabolite": pd.read_csv(source / "metabolite_paired_changes.csv", index_col=0),
    }
    if any(not blocks["Species"].index.equals(frame.index) for frame in blocks.values()):
        raise ValueError("Local omics matrices are not aligned")
    signs, exact = generate_signs(len(blocks["Species"]))
    if not exact or len(signs) != 256:
        raise RuntimeError("The local analysis must enumerate all 256 sign patterns")
    result = analyze_multiview(
        blocks,
        signs=signs,
        k_grid=tuple(config["k_grid"]),
        influence_constraint=False,
    )

    output = PROJECT / "results" / "local_application"
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, block_result in result.block_results.items():
        feature_table = block_result.feature_table.copy()
        if name == "Metabolite":
            metadata = pd.read_csv(source / "metabolite_annotation_metadata.csv")
            feature_table = feature_table.merge(metadata, on="feature", how="left")
        elif name == "KO":
            metadata = pd.read_csv(source / "ko_annotation_metadata.csv")
            feature_table = feature_table.merge(
                metadata, left_on="feature", right_on="KOEntry", how="left"
            ).drop(columns="KOEntry")
        feature_table.to_csv(
            output / f"{name.lower()}_stable_feature_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        block_result.aggregation_table.to_csv(
            output / f"{name.lower()}_aggregation_scales.csv",
            index=False,
            encoding="utf-8-sig",
        )
        block_result.permutation_table.to_csv(
            output / f"{name.lower()}_permutation_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary_rows.append(
            {
                "block": name,
                "subjects": len(blocks[name]),
                "features": blocks[name].shape[1],
                "block_exact_p": block_result.block_pvalue,
                "selected_k": block_result.selected_k,
                "maxT_fwer_lt_0_05": int((feature_table["maxT_fwer"] < 0.05).sum()),
                "feature_fdr_lt_0_05": int((feature_table["feature_fdr"] < 0.05).sum()),
                "feature_fdr_lt_0_10": int((feature_table["feature_fdr"] < 0.10).sum()),
                "eight_of_eight_direction": int(
                    (feature_table["direction_consistent_n"] == len(blocks[name])).sum()
                ),
            }
        )
    block_summary = pd.DataFrame(summary_rows)
    block_summary.to_csv(output / "block_summary.csv", index=False, encoding="utf-8-sig")
    result.joint_table.to_csv(
        output / "joint_intersection_union_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ko = result.block_results["KO"].feature_table
    metabolite = result.block_results["Metabolite"].feature_table
    targets = pd.concat(
        [
            ko.loc[ko["feature"].isin(["K00108", "K00130"])].assign(block="KO"),
            metabolite.loc[
                metabolite["feature"].str.contains("Betaine|3-Methyl-L-histidine", case=False)
            ].assign(block="Metabolite"),
        ],
        ignore_index=True,
    )
    targets.to_csv(output / "prespecified_feature_status.csv", index=False, encoding="utf-8-sig")

    checks = pd.DataFrame(
        [
            {
                "check": "all_blocks_aligned",
                "passed": all(blocks["Species"].index.equals(frame.index) for frame in blocks.values()),
            },
            {"check": "all_256_sign_patterns", "passed": len(signs) == 256},
            {
                "check": "joint_p_equals_largest_block_p",
                "passed": abs(
                    result.joint_pvalue - max(row["block_exact_p"] for row in summary_rows)
                )
                < 1e-12,
            },
            {
                "check": "joint_result_positive_at_0_05",
                "passed": result.joint_pvalue <= 0.05,
            },
        ]
    )
    checks.to_csv(output / "self_check.csv", index=False, encoding="utf-8-sig")
    run = {
        "method": config["method_name"],
        "abbreviation": config["method_abbreviation"],
        "joint_hypothesis": "stable paired location shift in every included omics block",
        "joint_iut_pvalue": result.joint_pvalue,
        "joint_evidence": result.joint_evidence,
        "sign_patterns": len(signs),
        "exact_enumeration": exact,
        "blocks": block_summary.to_dict(orient="records"),
    }
    (output / "run_summary.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, ensure_ascii=False, indent=2))
    print("\nLeading maxT-controlled features")
    for name, block_result in result.block_results.items():
        table = block_result.feature_table
        print(f"\n{name}")
        print(
            table.loc[table["maxT_fwer"] < 0.05]
            .head(10)[
                [
                    "feature",
                    "mean_delta",
                    "cohen_dz",
                    "direction_consistent_n",
                    "stable_score",
                    "maxT_fwer",
                ]
            ]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
