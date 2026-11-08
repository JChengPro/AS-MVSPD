#!/usr/bin/env python3
"""Prepare traceable source-data tables for publication figures.

This script never modifies source omics files. It derives uncertainty intervals,
additional benchmark results, stability analyses, and plotting tables from the
locked project inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from stablejoint import analyze_multiview, generate_signs  # noqa: E402
from stablejoint.benchmark import evaluate_methods, simulate_paired_blocks  # noqa: E402


RESULTS = PROJECT / "results"
SOURCE = RESULTS / "publication_figures" / "source_data"
TABLES = RESULTS / "publication_figures" / "tables"
LOCAL_DATA = PROJECT / "data" / "local"
PUBLIC_DATA = PROJECT / "data" / "public_prepared"


def save_csv(frame: pd.DataFrame, name: str, folder: Path = SOURCE) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / name, index=False, encoding="utf-8-sig")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_binary(
    frame: pd.DataFrame,
    groups: list[str],
    value: str = "rejected_0_05",
) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, dropna=False, sort=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        successes = int(part[value].astype(bool).sum())
        total = int(len(part))
        low, high = wilson_interval(successes, total)
        row = dict(zip(groups, key_tuple))
        row.update(
            {
                "successes": successes,
                "replicates": total,
                "rate": successes / total,
                "ci_low": low,
                "ci_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_existing_simulation() -> None:
    replicate = pd.read_csv(
        RESULTS / "simulation" / "replicate_results.csv", keep_default_na=False
    )
    summary = summarize_binary(
        replicate,
        ["n_subjects", "scenario", "method"],
    )
    save_csv(summary, "simulation_performance_with_ci.csv")

    recovery = pd.read_csv(
        RESULTS / "simulation" / "feature_recovery.csv", keep_default_na=False
    )
    recovery_summary = (
        recovery.groupby(
            ["n_subjects", "scenario", "selection_method", "block"], as_index=False
        )
        .agg(
            precision_mean=("precision", "mean"),
            precision_se=("precision", "sem"),
            recall_mean=("recall", "mean"),
            recall_se=("recall", "sem"),
            selected_n_median=("selected_n", "median"),
            replicates=("replicate", "nunique"),
        )
    )
    save_csv(recovery_summary, "simulation_feature_recovery_with_se.csv")


def run_extended_simulation(config: dict) -> None:
    seed = int(config["seed"]) + 6000
    rng = np.random.default_rng(seed)
    k_grid = tuple(int(x) for x in config["k_grid"])
    rows: list[dict] = []
    ablation_repetitions = 50
    sensitivity_repetitions = 500

    for n_subjects in (8, 12, 20):
        signs, _ = generate_signs(
            n_subjects,
            max_random=255 if n_subjects > 8 else 4095,
            seed=seed + n_subjects,
        )
        for scenario in ("stable_sparse", "stable_plus_outlier"):
            for replicate in range(ablation_repetitions):
                blocks, _ = simulate_paired_blocks(
                    n_subjects=n_subjects,
                    n_features=200,
                    n_signal=5,
                    scenario=scenario,
                    rng=rng,
                )
                main = analyze_multiview(
                    blocks, signs=signs, k_grid=k_grid, influence_constraint=False
                ).joint_pvalue
                loo = analyze_multiview(
                    blocks, signs=signs, k_grid=k_grid, influence_constraint=True
                ).joint_pvalue
                pvalues = {
                    "AS-MVSPD": main,
                    "AS-MVSPD+LOO": loo,
                }
                for k in (1, 5, 20):
                    pvalues[f"Fixed top-{k}"] = analyze_multiview(
                        blocks,
                        signs=signs,
                        k_grid=(k,),
                        influence_constraint=False,
                    ).joint_pvalue
                for method, pvalue in pvalues.items():
                    rows.append(
                        {
                            "analysis": "component_ablation",
                            "scenario": scenario,
                            "n_subjects": n_subjects,
                            "n_signal": 5,
                            "effect": 1.4,
                            "loading_scale": 0.35,
                            "replicate": replicate,
                            "method": method,
                            "pvalue": pvalue,
                            "rejected_0_05": pvalue <= 0.05,
                        }
                    )

    sensitivity_designs = []
    for effect in (0.8, 1.1, 1.4, 1.8):
        sensitivity_designs.append(("effect", 12, 5, effect, 0.35))
    for n_signal in (1, 5, 20, 50):
        sensitivity_designs.append(("sparsity", 12, n_signal, 1.4, 0.35))
    for loading_scale in (0.0, 0.35, 0.7):
        sensitivity_designs.append(("correlation", 12, 5, 1.4, loading_scale))

    for analysis, n_subjects, n_signal, effect, loading_scale in sensitivity_designs:
        signs, _ = generate_signs(n_subjects, max_random=255, seed=seed + 12)
        for replicate in range(sensitivity_repetitions):
            blocks, _ = simulate_paired_blocks(
                n_subjects=n_subjects,
                n_features=200,
                n_signal=n_signal,
                scenario="stable_sparse",
                rng=rng,
                effect=effect,
                loading_scale=loading_scale,
            )
            pvalues, _ = evaluate_methods(blocks, signs, k_grid=k_grid)
            for method, pvalue in pvalues.items():
                rows.append(
                    {
                        "analysis": analysis,
                        "scenario": "stable_sparse",
                        "n_subjects": n_subjects,
                        "n_signal": n_signal,
                        "effect": effect,
                        "loading_scale": loading_scale,
                        "replicate": replicate,
                        "method": method,
                        "pvalue": pvalue,
                        "rejected_0_05": pvalue <= 0.05,
                    }
                )

    replicate_frame = pd.DataFrame(rows)
    save_csv(replicate_frame, "simulation_extended_replicates.csv")
    summary = summarize_binary(
        replicate_frame,
        [
            "analysis",
            "scenario",
            "n_subjects",
            "n_signal",
            "effect",
            "loading_scale",
            "method",
        ],
    )
    save_csv(summary, "simulation_extended_with_ci.csv")


def load_public_dataset(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(folder / "microbiome_paired_changes.csv", index_col=0)
    y = pd.read_csv(folder / "metabolome_paired_changes.csv", index_col=0)
    pairs = pd.read_csv(folder / "pair_mapping.csv", keep_default_na=False)
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    if not x.index.equals(y.index):
        raise ValueError(f"Unaligned public data: {folder.name}")
    return x, y, pairs


def iter_public_subsets():
    from stablejoint.public_datasets import iter_public_subsets as current
    yield from current(PUBLIC_DATA)



def prepare_public_stability(config: dict) -> None:
    primary_seed = int(config["seed"])
    seed = primary_seed + 7000
    k_grid = tuple(int(x) for x in config["k_grid"])
    rng = np.random.default_rng(seed)
    overview_rows = []
    baseline_rows = []
    loo_rows = []
    resample_rows = []
    seed_rows = []

    for dataset_id, subset, label, x, y in iter_public_subsets():
        n_subjects = len(x)
        signs, exact = generate_signs(
            n_subjects,
            max_random=255 if n_subjects > 8 else 4095,
            seed=primary_seed + n_subjects,
        )
        blocks = {"Microbiome": x, "Metabolome": y}
        result = analyze_multiview(blocks, signs=signs, k_grid=k_grid)
        overview_rows.append(
            {
                "dataset_id": dataset_id,
                "subset": subset,
                "display_label": label,
                "paired_subjects": n_subjects,
                "time_points": 2,
                "microbiome_features": x.shape[1],
                "metabolome_features": y.shape[1],
                "sign_patterns": len(signs),
                "exact_enumeration": exact,
                "microbiome_p": result.block_results["Microbiome"].block_pvalue,
                "metabolome_p": result.block_results["Metabolome"].block_pvalue,
                "joint_p": result.joint_pvalue,
            }
        )

        pvalues, _ = evaluate_methods(blocks, signs, k_grid=k_grid)
        for method, pvalue in pvalues.items():
            baseline_rows.append(
                {
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "display_label": label,
                    "method": method,
                    "joint_p": pvalue,
                }
            )

        loo_signs, loo_exact = generate_signs(
            n_subjects - 1,
            max_random=255 if n_subjects - 1 > 8 else 4095,
            seed=seed + n_subjects - 1,
        )
        for omitted in x.index:
            keep = x.index != omitted
            loo_result = analyze_multiview(
                {"Microbiome": x.loc[keep], "Metabolome": y.loc[keep]},
                signs=loo_signs,
                k_grid=k_grid,
            )
            loo_rows.append(
                {
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "display_label": label,
                    "omitted_subject": omitted,
                    "joint_p": loo_result.joint_pvalue,
                    "microbiome_p": loo_result.block_results["Microbiome"].block_pvalue,
                    "metabolome_p": loo_result.block_results["Metabolome"].block_pvalue,
                    "sign_patterns": len(loo_signs),
                    "exact_enumeration": loo_exact,
                }
            )

        for replicate in range(30):
            sampled_blocks = {}
            for block_name, block in blocks.items():
                size = max(1, int(np.ceil(block.shape[1] * 0.8)))
                columns = rng.choice(block.columns.to_numpy(), size=size, replace=False)
                sampled_blocks[block_name] = block.loc[:, columns]
            sampled = analyze_multiview(sampled_blocks, signs=signs, k_grid=k_grid)
            for block_name, block_result in sampled.block_results.items():
                resample_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "subset": subset,
                        "display_label": label,
                        "replicate": replicate,
                        "result_level": block_name,
                        "pvalue": block_result.block_pvalue,
                        "selected_k": block_result.selected_k,
                    }
                )
            resample_rows.append(
                {
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "display_label": label,
                    "replicate": replicate,
                    "result_level": "Joint IUT",
                    "pvalue": sampled.joint_pvalue,
                    "selected_k": np.nan,
                }
            )

        seed_count = 1 if exact else 20
        for seed_index in range(seed_count):
            seed_signs, seed_exact = generate_signs(
                n_subjects,
                max_random=255 if n_subjects > 8 else 4095,
                seed=seed + 100 + seed_index,
            )
            seeded = analyze_multiview(blocks, signs=seed_signs, k_grid=k_grid)
            seed_rows.append(
                {
                    "dataset_id": dataset_id,
                    "subset": subset,
                    "display_label": label,
                    "seed_index": seed_index,
                    "joint_p": seeded.joint_pvalue,
                    "exact_enumeration": seed_exact,
                }
            )

    overview = pd.DataFrame(overview_rows)
    save_csv(overview, "public_dataset_overview.csv")
    save_csv(pd.DataFrame(baseline_rows), "public_method_comparison.csv")
    save_csv(pd.DataFrame(loo_rows), "public_leave_one_subject.csv")
    save_csv(pd.DataFrame(resample_rows), "public_feature_resampling.csv")
    save_csv(pd.DataFrame(seed_rows), "public_seed_sensitivity.csv")
    save_csv(overview, "Table_1_public_validation.csv", TABLES)


def prepare_method_validation() -> None:
    """Expose method-validation summaries through the figure source-data API."""

    validation = RESULTS / "method_validation"
    required = [
        "three_view_iut_summary.csv",
        "published_baseline_summary.csv",
        "influence_robustness_replicates.csv",
        "influence_robustness_summary.csv",
        "partial_null_fwer_summary.csv",
        "real_data_published_baselines.csv",
        "structured_correlation_global_summary.csv",
        "structured_correlation_fwer_summary.csv",
        "asymmetry_joint_summary.csv",
        "asymmetry_fwer_summary.csv",
    ]
    missing = [name for name in required if not (validation / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Run scripts/09_run_method_validation.py first; missing: "
            + ", ".join(missing)
        )
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(validation / name, SOURCE / name)


def raw_local_prepost(
    mapping: pd.DataFrame,
    selected_species: list[str],
    selected_ko: list[str],
    selected_met_ids: list[str],
) -> pd.DataFrame:
    audit = json.loads((LOCAL_DATA / "audit.json").read_text(encoding="utf-8"))
    paths = {key: Path(value) for key, value in audit["source_paths"].items()}
    pre_mg = mapping["metagenome_pre_id"].tolist()
    post_mg = mapping["metagenome_post_id"].tolist()
    pre_met = mapping["metabolome_pre_id"].tolist()
    post_met = mapping["metabolome_post_id"].tolist()
    animal_ids = mapping["animal_id"].astype(str).tolist()
    rows = []

    raw_species = pd.read_excel(paths["species"]).dropna(subset=["Species"])
    abundance = (
        raw_species.set_index("Species")[pre_mg + post_mg]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .T
    )
    abundance = abundance.groupby(level=0, axis=1).sum()
    abundance = abundance.div(abundance.sum(axis=1), axis=0) * 1_000_000.0
    retained_species = pd.read_csv(LOCAL_DATA / "species_paired_changes.csv", index_col=0).columns
    values = abundance.loc[:, retained_species].to_numpy(dtype=float) + 0.5
    values = values / values.sum(axis=1, keepdims=True) * 1_000_000.0
    clr = np.log(values) - np.log(values).mean(axis=1, keepdims=True)
    clr = pd.DataFrame(clr, index=abundance.index, columns=retained_species)
    for feature in selected_species:
        for time, sample_ids in (("Pre", pre_mg), ("Post", post_mg)):
            for animal, sample in zip(animal_ids, sample_ids):
                rows.append(
                    {
                        "animal_id": animal,
                        "block": "Species",
                        "feature": feature,
                        "display_label": feature.replace("s__", "").replace("_", " "),
                        "time": time,
                        "value": float(clr.loc[sample, feature]),
                        "scale": "CLR abundance",
                    }
                )

    raw_ko = pd.read_excel(paths["ko"])
    numeric = raw_ko[pre_mg + post_mg].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ppm = numeric.div(numeric.sum(axis=0), axis=1) * 1_000_000.0
    ko_abundance = ppm.loc[raw_ko["KOEntry"].notna()].set_axis(
        raw_ko.loc[raw_ko["KOEntry"].notna(), "KOEntry"]
    ).T
    ko_abundance = ko_abundance.groupby(level=0, axis=1).sum()
    ko_log = np.log2(ko_abundance + 0.5)
    ko_meta = pd.read_csv(LOCAL_DATA / "ko_annotation_metadata.csv", keep_default_na=False)
    descriptions = ko_meta.set_index("KOEntry")["ko_description"].to_dict()
    for feature in selected_ko:
        description = str(descriptions.get(feature, "")).strip()
        label = f"{feature} ({description})" if description else feature
        for time, sample_ids in (("Pre", pre_mg), ("Post", post_mg)):
            for animal, sample in zip(animal_ids, sample_ids):
                rows.append(
                    {
                        "animal_id": animal,
                        "block": "KO",
                        "feature": feature,
                        "display_label": label,
                        "time": time,
                        "value": float(ko_log.loc[sample, feature]),
                        "scale": "log2(ppm + 0.5)",
                    }
                )

    metabolite_meta = pd.read_csv(
        LOCAL_DATA / "metabolite_annotation_metadata.csv", keep_default_na=False
    )
    selected_meta = metabolite_meta.loc[metabolite_meta["Met ID"].isin(selected_met_ids)]
    raw_modes = []
    for ion_mode in ("pos", "neg"):
        frame = pd.read_excel(paths["metabolite"], sheet_name=ion_mode)
        frame["ion_mode"] = ion_mode
        for sample in pre_met + post_met:
            frame[sample] = pd.to_numeric(frame[sample], errors="coerce")
            total = frame[sample].sum()
            frame[sample] = frame[sample] / total * 1_000_000.0
        raw_modes.append(frame)
    raw_met = pd.concat(raw_modes, ignore_index=True)
    for _, meta in selected_meta.iterrows():
        met_id = str(meta["Met ID"])
        ion_mode = str(meta["ion_mode"])
        feature = str(meta["feature"])
        match = raw_met.loc[raw_met["Met ID"].astype(str).eq(met_id) & raw_met["ion_mode"].eq(ion_mode)]
        if len(match) != 1:
            raise ValueError(f"Expected one raw metabolite row for {met_id}/{ion_mode}")
        match = match.iloc[0]
        name = str(meta["metabolite_name_clean"])
        label = met_id if not name else f"{met_id} (putative {name})"
        for time, sample_ids in (("Pre", pre_met), ("Post", post_met)):
            for animal, sample in zip(animal_ids, sample_ids):
                rows.append(
                    {
                        "animal_id": animal,
                        "block": "Metabolite",
                        "feature": feature,
                        "display_label": label,
                        "time": time,
                        "value": float(np.log2(float(match[sample]) + 1.0)),
                        "scale": "log2(sum-normalized intensity + 1)",
                    }
                )
    return pd.DataFrame(rows)


def prepare_local_data() -> None:
    mapping = pd.read_csv(LOCAL_DATA / "sample_mapping.csv", keep_default_na=False)
    block_summary = pd.read_csv(
        RESULTS / "local_application" / "block_summary.csv", keep_default_na=False
    )
    save_csv(block_summary, "local_block_summary.csv")
    save_csv(block_summary, "Table_2_local_block_tests.csv", TABLES)

    frames = {}
    deltas = {}
    for block, stem in (("Species", "species"), ("KO", "ko"), ("Metabolite", "metabolite")):
        frames[block] = pd.read_csv(
            RESULTS / "local_application" / f"{stem}_stable_feature_statistics.csv",
            keep_default_na=False,
        )
        deltas[block] = pd.read_csv(
            LOCAL_DATA / f"{stem}_paired_changes.csv", index_col=0
        )

    selected_species = frames["Species"].loc[frames["Species"]["maxT_fwer"] < 0.05, "feature"].tolist()
    selected_ko = frames["KO"].loc[frames["KO"]["maxT_fwer"] < 0.05, "feature"].tolist()
    selected_met = frames["Metabolite"].loc[
        frames["Metabolite"]["maxT_fwer"] < 0.05
    ].sort_values("stable_rank").head(6)
    selected_met_features = selected_met["feature"].tolist()
    selected_met_ids = selected_met["Met ID"].astype(str).tolist()

    selected_rows = []
    for block, features in (
        ("Species", selected_species),
        ("KO", selected_ko),
        ("Metabolite", selected_met_features),
    ):
        part = frames[block].loc[frames[block]["feature"].isin(features)].copy()
        part["block"] = block
        if block == "Species":
            part["display_label"] = part["feature"].str.replace("s__", "", regex=False).str.replace("_", " ", regex=False)
        elif block == "KO":
            part["display_label"] = part["feature"] + " (" + part["ko_description"].astype(str) + ")"
        else:
            part["display_label"] = part["Met ID"].astype(str)
        part["signed_stable_score"] = np.sign(part["mean_delta"]) * part["stable_score"]
        selected_rows.append(part)
    selected = pd.concat(selected_rows, ignore_index=True)
    save_csv(selected, "local_representative_features.csv")
    table_columns = [
        "block",
        "feature",
        "display_label",
        "mean_delta",
        "cohen_dz",
        "direction_consistent_n",
        "stable_score",
        "maxT_fwer",
        "stable_rank",
    ]
    save_csv(selected[table_columns], "Table_3_representative_features.csv", TABLES)

    heatmap_rows = []
    for row in selected.itertuples(index=False):
        block = row.block
        values = deltas[block][row.feature].astype(float)
        sd = float(values.std(ddof=1))
        scaled = values / sd if sd > 0 else values * 0.0
        for animal, value, standardized in zip(values.index, values, scaled):
            heatmap_rows.append(
                {
                    "animal_id": str(animal),
                    "block": block,
                    "feature": row.feature,
                    "display_label": row.display_label,
                    "paired_change": float(value),
                    "sd_scaled_change": float(standardized),
                }
            )
    save_csv(pd.DataFrame(heatmap_rows), "local_representative_heatmap.csv")

    representative_prepost = raw_local_prepost(
        mapping,
        selected_species=[selected_species[0]],
        selected_ko=selected_ko,
        selected_met_ids=["M109T115_pos"],
    )
    save_csv(representative_prepost, "local_representative_prepost.csv")

    count_rows = []
    all_fwer_rows = []
    for block, frame in frames.items():
        passed = frame.loc[frame["maxT_fwer"] < 0.05].copy()
        count_rows.append(
            {
                "block": block,
                "up": int((passed["mean_delta"] > 0).sum()),
                "down": int((passed["mean_delta"] < 0).sum()),
                "total": int(len(passed)),
            }
        )
        passed["block"] = block
        if block == "Species":
            passed["display_label"] = passed["feature"].str.replace("s__", "", regex=False).str.replace("_", " ", regex=False)
            passed["annotation_status"] = "species name"
        elif block == "KO":
            passed["display_label"] = passed["feature"]
            passed["annotation_status"] = "KEGG orthology"
        else:
            passed["display_label"] = passed["Met ID"].astype(str)
            hmdb = passed["HMDB ID"].astype(str).str.len().gt(0)
            kegg = passed["KEGG ID"].astype(str).str.len().gt(0)
            passed["annotation_status"] = np.select(
                [hmdb & kegg, hmdb, kegg],
                ["HMDB + KEGG", "HMDB only", "KEGG only"],
                default="database name only",
            )
        all_fwer_rows.append(passed)
    save_csv(pd.DataFrame(count_rows), "local_fwer_direction_counts.csv")
    all_fwer = pd.concat(all_fwer_rows, ignore_index=True)
    save_csv(all_fwer, "local_all_maxT_features.csv")

    full_rank = []
    for block, frame in frames.items():
        part = frame[["feature", "mean_delta", "stable_score", "stable_rank", "maxT_fwer"]].copy()
        part["block"] = block
        part["signed_stable_score"] = np.sign(part["mean_delta"]) * part["stable_score"]
        if block == "Metabolite":
            part = part.merge(frame[["feature", "Met ID"]], on="feature", how="left")
        full_rank.append(part)
    save_csv(pd.concat(full_rank, ignore_index=True), "local_all_feature_ranks.csv")

    metabolite_pass = frames["Metabolite"].loc[frames["Metabolite"]["maxT_fwer"] < 0.05]
    full_met_heat = []
    for row in metabolite_pass.sort_values("stable_rank").itertuples(index=False):
        values = deltas["Metabolite"][row.feature].astype(float)
        sd = float(values.std(ddof=1))
        for animal, value in values.items():
            full_met_heat.append(
                {
                    "animal_id": str(animal),
                    "feature": row.feature,
                    "met_id": getattr(row, "_12", "") if hasattr(row, "_12") else "",
                    "stable_rank": int(row.stable_rank),
                    "sd_scaled_change": float(value / sd) if sd > 0 else 0.0,
                }
            )
    met_id_map = frames["Metabolite"].set_index("feature")["Met ID"].astype(str).to_dict()
    full_met_heat_frame = pd.DataFrame(full_met_heat)
    full_met_heat_frame["met_id"] = full_met_heat_frame["feature"].map(met_id_map)
    save_csv(full_met_heat_frame, "local_all_metabolite_heatmap.csv")

    loo_rows = []
    for row in selected.itertuples(index=False):
        values = deltas[row.block][row.feature].astype(float)
        for omitted in values.index:
            remaining = values.drop(index=omitted)
            sd = float(remaining.std(ddof=1))
            loo_rows.append(
                {
                    "block": row.block,
                    "feature": row.feature,
                    "display_label": row.display_label,
                    "omitted_animal": str(omitted),
                    "loo_cohen_dz": float(remaining.mean() / sd) if sd > 0 else 0.0,
                }
            )
    save_csv(pd.DataFrame(loo_rows), "local_leave_one_effects.csv")

    for block, stem in (("Species", "species"), ("KO", "ko"), ("Metabolite", "metabolite")):
        scale = pd.read_csv(
            RESULTS / "local_application" / f"{stem}_aggregation_scales.csv",
            keep_default_na=False,
        )
        scale["block"] = block
        save_csv(scale, f"local_{stem}_aggregation_scales.csv")


def prepare_qc_data() -> None:
    mapping = pd.read_csv(LOCAL_DATA / "sample_mapping.csv", keep_default_na=False)
    mapping_columns = [
        "animal_id",
        "metagenome_pre_id",
        "metagenome_post_id",
        "metabolome_pre_id",
        "metabolome_post_id",
        "metagenome_pair_present",
        "metabolome_pair_present",
        "cross_omics_complete",
    ]
    save_csv(mapping.loc[:, mapping_columns].copy(), "qc_sample_mapping.csv")

    species_qc = pd.read_csv(LOCAL_DATA / "species_feature_qc.csv", keep_default_na=False)
    metabolite_qc = pd.read_csv(
        LOCAL_DATA / "metabolite_feature_qc.csv", keep_default_na=False
    )
    final_met_ids = set(
        pd.read_csv(LOCAL_DATA / "metabolite_annotation_metadata.csv", keep_default_na=False)[
            "Met ID"
        ].astype(str)
    )
    metabolite_qc["retained_final"] = metabolite_qc["Met ID"].astype(str).isin(final_met_ids)
    save_csv(metabolite_qc, "qc_metabolite_features.csv")
    flow = pd.DataFrame(
        [
            {"block": "Species", "stage": "Input", "features": len(species_qc), "stage_order": 1},
            {
                "block": "Species",
                "stage": "Retained",
                "features": int(as_bool(species_qc["retained_before_mad"]).sum()),
                "stage_order": 2,
            },
            {"block": "KO", "stage": "Input", "features": 5476, "stage_order": 1},
            {"block": "KO", "stage": "Annotated", "features": 5475, "stage_order": 2},
            {
                "block": "Metabolite",
                "stage": "Input",
                "features": len(metabolite_qc),
                "stage_order": 1,
            },
            {
                "block": "Metabolite",
                "stage": "QC pass",
                "features": int(as_bool(metabolite_qc["qc_pass"]).sum()),
                "stage_order": 2,
            },
            {
                "block": "Metabolite",
                "stage": "QC + annotation",
                "features": int(
                    (
                        as_bool(metabolite_qc["qc_pass"])
                        & as_bool(metabolite_qc["annotation_pass"])
                    ).sum()
                ),
                "stage_order": 3,
            },
            {
                "block": "Metabolite",
                "stage": "Deduplicated",
                "features": int(metabolite_qc["retained_final"].sum()),
                "stage_order": 4,
            },
        ]
    )
    save_csv(flow, "qc_feature_filtering.csv")

    distribution_rows = []
    subject_rows = []
    for block, stem in (("Species", "species"), ("KO", "ko"), ("Metabolite", "metabolite")):
        delta = pd.read_csv(LOCAL_DATA / f"{stem}_paired_changes.csv", index_col=0)
        for feature, mean_value in delta.mean(axis=0).items():
            distribution_rows.append(
                {"block": block, "feature": feature, "mean_paired_change": float(mean_value)}
            )
        sd = delta.std(axis=0, ddof=1).replace(0.0, np.nan)
        standardized = delta.div(sd, axis=1)
        for animal, values in standardized.iterrows():
            subject_rows.append(
                {
                    "animal_id": str(animal),
                    "block": block,
                    "median_abs_sd_scaled_change": float(values.abs().median()),
                }
            )
    save_csv(pd.DataFrame(distribution_rows), "qc_feature_change_distributions.csv")
    save_csv(pd.DataFrame(subject_rows), "qc_subject_change_magnitude.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-expensive",
        action="store_true",
        help="rerun extended simulations and public resampling analyses",
    )
    parser.add_argument('--section',choices=['all','simulation','extended','validation','public','local'],default='all')
    args = parser.parse_args()
    config = json.loads((PROJECT / 'config/experiment.json').read_text())
    SOURCE.mkdir(parents=True,exist_ok=True)
    if args.section in ['all','simulation']:prepare_existing_simulation()
    if args.section in ['all','extended']:
        if args.refresh_expensive or not (SOURCE/'simulation_extended_with_ci.csv').exists():run_extended_simulation(config)
    if args.section in ['all','public']:
        import subprocess
        subprocess.run([sys.executable,str(PROJECT/'scripts/03_run_public_validation.py'),'--sensitivity'],check=True)
        for f in (PROJECT/'results/public_validation').glob('*.csv'):
            if not f.name.endswith('_features.csv'):shutil.copy2(f,SOURCE/f.name)
    if args.section in ['all','validation']:prepare_method_validation()
    if args.section in ['all','local']:
        prepare_local_data();prepare_qc_data()
    print(f'Prepared {args.section} figure tables in {SOURCE}')

if __name__ == '__main__':main()
