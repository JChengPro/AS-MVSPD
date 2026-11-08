#!/usr/bin/env python3
"""Reanalyse the local data after sample-wise CLR transformation of KO abundances."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent

sys.path.insert(0, str(PROJECT))

from stablejoint import analyze_multiview, generate_signs  # noqa: E402


PSEUDOCOUNT_PPM = 0.5


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def matrix_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def build_ko_clr_changes() -> tuple[pd.DataFrame, dict[str, object]]:
    from stablejoint import local_preprocessing as core
    from stablejoint.local_inputs import configured_paths
    paths = configured_paths(PROJECT)
    mapping = core.load_sample_map(paths.sample_map)

    raw = pd.read_excel(paths.ko)
    pre_ids = mapping["metagenome_pre_id"].astype(str).tolist()
    post_ids = mapping["metagenome_post_id"].astype(str).tolist()
    sample_ids = pre_ids + post_ids
    numeric = raw[sample_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    totals = numeric.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("KO input contains a non-positive sample total")

    ppm = numeric.div(totals, axis=1) * 1_000_000.0
    annotated = raw["KOEntry"].notna()
    abundance = ppm.loc[annotated].set_axis(raw.loc[annotated, "KOEntry"]).T
    abundance = abundance.groupby(level=0, axis=1).sum()
    log_abundance = np.log(abundance + PSEUDOCOUNT_PPM)
    clr = log_abundance.sub(log_abundance.mean(axis=1), axis=0)

    pre = clr.loc[pre_ids].copy()
    post = clr.loc[post_ids].copy()
    animal_ids = mapping["animal_id"].astype(str).to_numpy()
    pre.index = animal_ids
    post.index = animal_ids
    delta = post - pre
    delta.index.name = "animal_id"

    audit = {
        "source_ko": str(paths.ko),
        "subjects": int(len(mapping)),
        "features": int(delta.shape[1]),
        "pseudocount_ppm": PSEUDOCOUNT_PPM,
        "transform": "sample-wise CLR of KO ppm, followed by Post minus Pre",
    }
    return delta, audit


def main() -> None:
    config = json.loads((PROJECT / "config" / "experiment.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    local = PROJECT / "data" / "local"
    output = PROJECT / "results" / "local_application" / "ko_clr_sensitivity"
    output.mkdir(parents=True, exist_ok=True)

    ko_clr, audit = build_ko_clr_changes()
    species = pd.read_csv(local / "species_paired_changes.csv", index_col=0)
    metabolite = pd.read_csv(local / "metabolite_paired_changes.csv", index_col=0)
    ko_primary = pd.read_csv(local / "ko_paired_changes.csv", index_col=0)
    for frame in (species, metabolite, ko_primary, ko_clr):
        frame.index = frame.index.astype(str)
    if not species.index.equals(ko_clr.index) or not species.index.equals(metabolite.index):
        raise ValueError("Local multi-view subject order is not aligned")
    if not ko_primary.columns.equals(ko_clr.columns):
        raise ValueError("Primary and CLR KO feature sets are not identical")

    signs, exact = generate_signs(
        len(ko_clr), max_random=int(config["max_random_sign_flips"]), seed=seed
    )
    primary = analyze_multiview(
        {"Species": species, "KO": ko_primary, "Metabolite": metabolite},
        signs=signs,
        k_grid=tuple(config["k_grid"]),
    )
    sensitivity = analyze_multiview(
        {"Species": species, "KO": ko_clr, "Metabolite": metabolite},
        signs=signs,
        k_grid=tuple(config["k_grid"]),
    )

    rows = []
    for analysis_name, result in (("Primary KO log-ratio", primary), ("KO CLR sensitivity", sensitivity)):
        for view, block in result.block_results.items():
            rows.append(
                {
                    "analysis": analysis_name,
                    "view": view,
                    "view_p": block.block_pvalue,
                    "selected_k": block.selected_k,
                    "maxT_fwer_lt_0_05": int((block.feature_table["maxT_fwer"] < 0.05).sum()),
                    "bh_adjusted_p_lt_0_05": int((block.feature_table["feature_fdr"] < 0.05).sum()),
                    "bh_adjusted_p_lt_0_10": int((block.feature_table["feature_fdr"] < 0.10).sum()),
                }
            )
        rows.append(
            {
                "analysis": analysis_name,
                "view": "Joint IUT",
                "view_p": result.joint_pvalue,
                "selected_k": np.nan,
                "maxT_fwer_lt_0_05": np.nan,
                "bh_adjusted_p_lt_0_05": np.nan,
                "bh_adjusted_p_lt_0_10": np.nan,
            }
        )
    summary = pd.DataFrame(rows)

    primary_table = primary.block_results["KO"].feature_table.set_index("feature")
    clr_table = sensitivity.block_results["KO"].feature_table.set_index("feature")
    clr_aligned = clr_table.reindex(primary_table.index)
    if clr_aligned["stable_rank"].isna().any():
        raise ValueError("CLR KO results cannot be aligned to primary KO features")
    rank_rho = float(
        spearmanr(primary_table["stable_rank"], clr_aligned["stable_rank"]).statistic
    )
    direction_concordance = float(
        np.mean(
            np.sign(primary_table["mean_delta"].to_numpy())
            == np.sign(clr_aligned["mean_delta"].to_numpy())
        )
    )
    top20_primary = set(primary_table.nsmallest(20, "stable_rank").index)
    top20_clr = set(clr_table.nsmallest(20, "stable_rank").index)
    maxT_primary = set(primary_table.index[primary_table["maxT_fwer"] < 0.05])
    maxT_clr = set(clr_table.index[clr_table["maxT_fwer"] < 0.05])
    overlap = pd.DataFrame(
        [
            {
                "rank_spearman_all_features": rank_rho,
                "mean_direction_concordance": direction_concordance,
                "top20_overlap_n": len(top20_primary & top20_clr),
                "top20_jaccard": len(top20_primary & top20_clr) / len(top20_primary | top20_clr),
                "primary_maxT_n": len(maxT_primary),
                "clr_maxT_n": len(maxT_clr),
                "maxT_overlap_n": len(maxT_primary & maxT_clr),
                "maxT_jaccard": (
                    len(maxT_primary & maxT_clr) / len(maxT_primary | maxT_clr)
                    if maxT_primary | maxT_clr
                    else 1.0
                ),
            }
        ]
    )

    ko_clr.to_csv(output / "ko_clr_paired_changes.csv", encoding="utf-8-sig")
    summary.to_csv(output / "analysis_summary.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(output / "feature_stability_summary.csv", index=False, encoding="utf-8-sig")
    clr_table.reset_index().to_csv(output / "ko_clr_feature_results.csv", index=False, encoding="utf-8-sig")
    audit.update(
        {
            "seed": seed,
            "sign_patterns": int(len(signs)),
            "complete_sign_group": bool(exact),
            "sign_matrix_sha256": matrix_sha256(signs),
        }
    )
    (output / "run_metadata.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
