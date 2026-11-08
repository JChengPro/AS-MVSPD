#!/usr/bin/env python3
"""Build and validate the point-level data audit for Figure 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
AUTHORITATIVE = PROJECT / "results" / "method_validation"
FIGURE_SOURCE = PROJECT / "results" / "publication_figures" / "source_data"
AUDIT = PROJECT / "results" / "publication_figures" / "audit"

METHODS = ["AS-MVSPD", "Paired-aSPU-IUT", "Paired-CQ-IUT", "Paired-RankCOM-IUT"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()




def append_rows(
    rows: list[dict],
    frame: pd.DataFrame,
    panel: str,
    source_name: str,
    x_column: str,
    label_columns: list[str],
) -> None:
    digest = sha256(AUTHORITATIVE / source_name)
    for _, record in frame.iterrows():
        label = "; ".join(f"{column}={record[column]}" for column in label_columns)
        rows.append(
            {
                "panel": panel,
                "method_scenario": label,
                "x_value": record[x_column],
                "estimate": record["rate"],
                "wilson_ci_low": record["ci_low"],
                "wilson_ci_high": record["ci_high"],
                "replicate_count": int(record["replicates"]),
                "successes": int(record["successes"]),
                "authoritative_source": source_name,
                "source_sha256": digest,
            }
        )


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data',type=Path,default=PROJECT/'figures/source_data')
    args=parser.parse_args()
    global AUTHORITATIVE,FIGURE_SOURCE
    AUTHORITATIVE=args.source_data
    FIGURE_SOURCE=args.source_data
    source_names = [
        "three_view_iut_summary.csv",
        "published_baseline_summary.csv",
        "partial_null_fwer_summary.csv",
        "structured_correlation_global_summary.csv",
        "asymmetry_fwer_summary.csv",
    ]
    provenance = []
    for name in source_names:
        authoritative = AUTHORITATIVE / name
        copied = FIGURE_SOURCE / name
        same = sha256(authoritative) == sha256(copied)
        provenance.append(
            {
                "source": name,
                "authoritative_sha256": sha256(authoritative),
                "figure_source_sha256": sha256(copied),
                "identical": same,
            }
        )
        if not same:
            raise RuntimeError(f"Figure source is stale: {name}")

    rows: list[dict] = []
    three = pd.read_csv(AUTHORITATIVE / source_names[0])
    three = three.loc[three["method"].isin(METHODS)].sort_values(["method", "active_views"])
    append_rows(rows, three, "A", source_names[0], "active_views", ["method", "scenario"])

    baseline = pd.read_csv(AUTHORITATIVE / source_names[1])
    for n_subjects, panel in ((8, "B"), (12, "C")):
        part = baseline.loc[
            baseline["n_subjects"].eq(n_subjects) & baseline["method"].isin(METHODS)
        ].sort_values(["method", "n_signal"])
        append_rows(rows, part, panel, source_names[1], "n_signal", ["method", "n_subjects"])

    fwer = pd.read_csv(AUTHORITATIVE / source_names[2])
    part = fwer.loc[
        fwer["method"].eq("AS-MVSPD")
        & fwer["distribution"].isin(["normal", "t3", "zero_inflated", "heteroscedastic", "clr"])
    ].sort_values(["distribution", "n_signal"])
    append_rows(rows, part, "D", source_names[2], "n_signal", ["method", "distribution"])

    correlation = pd.read_csv(AUTHORITATIVE / source_names[3])
    part = correlation.loc[
        correlation["n_subjects"].isin([8, 12])
        & correlation["structure"].isin(["ar1", "block"])
        & correlation["scenario"].eq("stable_sparse")
        & correlation["method"].eq("AS-MVSPD")
    ].sort_values(["n_subjects", "structure", "rho"])
    append_rows(rows, part, "E", source_names[3], "rho", ["method", "n_subjects", "structure", "scenario"])

    asymmetry = pd.read_csv(AUTHORITATIVE / source_names[4])
    part = asymmetry.loc[asymmetry["n_signal"].isin([0, 5, 20])].sort_values(["n_signal", "g"])
    part = part.assign(method="AS-MVSPD")
    append_rows(rows, part, "F", source_names[4], "g", ["method", "n_subject"] if "n_subject" in part else ["method", "n_subjects", "n_signal"])

    audit = pd.DataFrame(rows)
    expected_counts = {"A": 16, "B": 12, "C": 12, "D": 15, "E": 16, "F": 18}
    actual_counts = audit.groupby("panel").size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected Figure 2 point counts: {actual_counts}")
    if set(audit["replicate_count"]) != {500}:
        raise RuntimeError(f"Figure 2 contains non-500 replicate counts: {sorted(set(audit['replicate_count']))}")

    AUDIT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT / "Figure2_data_audit.csv", index=False, encoding="utf-8-sig")
    report = {
        "authoritative_source_identity": provenance,
        "point_counts": actual_counts,
        "replicate_counts": sorted(set(audit["replicate_count"])),

        "numeric_inconsistencies": [],
    }
    (AUDIT / "Figure2_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(audit.groupby("panel").agg(points=("estimate", "size"), replicates=("replicate_count", "unique")))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
