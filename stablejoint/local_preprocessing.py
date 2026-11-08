"""Species and metabolite preprocessing for the paired rat study."""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from typing import Tuple, Sequence

import itertools

import numpy as np

import pandas as pd

N_RATS = 8

PSEUDOCOUNT_PPM = 0.5

SPECIES_PREVALENCE = 0.75

SPECIES_MEAN_PPM = 100.0

METABOLITE_SCORE = 0.8

METABOLITE_QC_CV = 30.0

KEEP_GRID = (2, 3, 5, 8, 10, 12)

@dataclass(frozen=True)
class Paths:
    project_root: Path
    analysis_root: Path
    results: Path
    sample_map: Path
    species: Path
    ko: Path
    metabolite: Path
    taxfun: Path

def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out

def exact_signflip_pvalues(delta: pd.DataFrame) -> pd.DataFrame:
    """Two-sided exact paired location test over all 2^8 sign patterns."""
    arr = delta.to_numpy(dtype=float)
    n = arr.shape[0]
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
    observed = np.abs(arr.mean(axis=0))
    permuted = np.abs((signs @ arr) / n)
    p = (permuted >= observed[None, :] - 1e-12).mean(axis=0)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1)
    dz = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    nonzero = np.sum(np.abs(arr) > 1e-12, axis=0)
    agreement = np.maximum(np.sum(arr > 0, axis=0), np.sum(arr < 0, axis=0))
    direction_consistency = np.divide(
        agreement,
        nonzero,
        out=np.zeros_like(mean),
        where=nonzero > 0,
    )
    result = pd.DataFrame(
        {
            "feature": delta.columns,
            "mean_delta": mean,
            "median_delta": np.median(arr, axis=0),
            "sd_delta": sd,
            "cohen_dz": dz,
            "direction_consistency": direction_consistency,
            "exact_signflip_p": p,
        }
    )
    result["signflip_fdr"] = bh_adjust(result["exact_signflip_p"])
    return result

def clr_transform(ppm_samples_by_features: pd.DataFrame, pseudocount: float = PSEUDOCOUNT_PPM) -> pd.DataFrame:
    values = ppm_samples_by_features.to_numpy(dtype=float)
    if np.any(values < 0):
        raise ValueError("Compositional matrix contains negative values")
    values = values + pseudocount
    values = values / values.sum(axis=1, keepdims=True) * 1_000_000.0
    log_values = np.log(values)
    clr = log_values - log_values.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=ppm_samples_by_features.index, columns=ppm_samples_by_features.columns)

def load_sample_map(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    mapping = mapping.loc[mapping["included_in_cross_omics_analysis"].astype(bool)].copy()
    mapping = mapping.sort_values("animal_id").reset_index(drop=True)
    if len(mapping) != N_RATS:
        raise ValueError(f"Expected {N_RATS} cross-omics rats, found {len(mapping)}")
    if mapping["species"].astype(str).str.lower().ne("rat").any():
        raise ValueError("The cross-omics mapping is not uniformly labelled as Rat")
    if mapping["animal_id"].duplicated().any():
        raise ValueError("Animal IDs are not unique")
    required = [
        "metagenome_pre_id",
        "metagenome_post_id",
        "metabolome_pre_id",
        "metabolome_post_id",
    ]
    if mapping[required].isna().any().any():
        raise ValueError("Incomplete cross-omics sample mapping")
    return mapping

def is_informative_species(name: str) -> bool:
    lower = str(name).lower()
    blocked = (
        "unclassified",
        "unidentified",
    )
    return str(name).startswith("s__") and not any(token in lower for token in blocked)

def load_species(paths: Paths, mapping: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(paths.species)
    raw = raw.dropna(subset=["Species"]).copy()
    pre_ids = mapping["metagenome_pre_id"].tolist()
    post_ids = mapping["metagenome_post_id"].tolist()
    sample_ids = pre_ids + post_ids
    missing = sorted(set(sample_ids) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing species samples: {missing}")
    abundance = raw.set_index("Species")[sample_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    abundance = abundance.groupby(level=0, axis=1).sum()
    abundance = abundance.div(abundance.sum(axis=1), axis=0) * 1_000_000.0
    prevalence = (abundance > 0).mean(axis=0)
    mean_ppm = abundance.mean(axis=0)
    informative = pd.Series([is_informative_species(x) for x in abundance.columns], index=abundance.columns)
    keep = (prevalence >= SPECIES_PREVALENCE) & (mean_ppm >= SPECIES_MEAN_PPM) & informative
    filtered = abundance.loc[:, keep]
    if filtered.shape[1] < max(KEEP_GRID):
        raise ValueError("Too few species remain after filtering")
    clr = clr_transform(filtered)
    delta = clr.loc[post_ids].set_axis(mapping["animal_id"], axis=0) - clr.loc[pre_ids].set_axis(mapping["animal_id"], axis=0)
    delta.index.name = "animal_id"
    feature_qc = pd.DataFrame(
        {
            "feature": abundance.columns,
            "prevalence": prevalence.values,
            "mean_ppm": mean_ppm.values,
            "informative_name": informative.values,
            "retained_before_mad": keep.values,
        }
    )
    raw_summary = pd.DataFrame(
        {
            "feature": filtered.columns,
            "pre_mean_ppm": abundance.loc[pre_ids, filtered.columns].mean(axis=0).values,
            "post_mean_ppm": abundance.loc[post_ids, filtered.columns].mean(axis=0).values,
        }
    )
    return delta, feature_qc, raw_summary

def _clean_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "-"} else text

def load_metabolites(
    paths: Paths, mapping: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for ion_mode in ("pos", "neg"):
        frame = pd.read_excel(paths.metabolite, sheet_name=ion_mode)
        frame["ion_mode"] = ion_mode
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    pre_ids = mapping["metabolome_pre_id"].tolist()
    post_ids = mapping["metabolome_post_id"].tolist()
    sample_ids = pre_ids + post_ids
    qc_ids = ["QC-1", "QC-2", "QC-3"]
    missing = sorted(set(sample_ids + qc_ids) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing metabolite samples: {missing}")
    for col in sample_ids + qc_ids + ["Score"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    normalization_rows = []
    for ion_mode in ("pos", "neg"):
        mode_mask = raw["ion_mode"].eq(ion_mode)
        for sample_id in sample_ids + qc_ids:
            raw_total = float(raw.loc[mode_mask, sample_id].sum())
            if not np.isfinite(raw_total) or raw_total <= 0:
                raise ValueError(f"Invalid total metabolite intensity for {ion_mode}/{sample_id}")
            factor = 1_000_000.0 / raw_total
            raw.loc[mode_mask, sample_id] = raw.loc[mode_mask, sample_id] * factor
            normalization_rows.append(
                {
                    "ion_mode": ion_mode,
                    "sample_id": sample_id,
                    "raw_total_intensity": raw_total,
                    "sum_normalization_factor": factor,
                    "normalized_total": float(raw.loc[mode_mask, sample_id].sum()),
                }
            )
    qc_mean = raw[qc_ids].mean(axis=1)
    qc_sd = raw[qc_ids].std(axis=1, ddof=1)
    raw["qc_cv_percent"] = np.where(qc_mean > 0, qc_sd / qc_mean * 100.0, np.inf)
    raw["metabolite_name_clean"] = raw["Metabolite Name"].map(_clean_identifier)
    raw["hmdb_clean"] = raw["HMDB ID"].map(_clean_identifier)
    raw["kegg_clean"] = raw["KEGG ID"].map(_clean_identifier)
    raw["annotation_id"] = np.where(
        raw["hmdb_clean"].ne(""),
        "HMDB:" + raw["hmdb_clean"],
        np.where(raw["kegg_clean"].ne(""), "KEGG:" + raw["kegg_clean"], ""),
    )
    qc_pass = (
        raw[qc_ids].notna().all(axis=1)
        & np.isfinite(raw["qc_cv_percent"])
        & raw["qc_cv_percent"].le(METABOLITE_QC_CV)
    )
    annotation_pass = (
        raw["metabolite_name_clean"].ne("")
        & raw["annotation_id"].ne("")
        & raw["Score"].ge(METABOLITE_SCORE)
    )
    raw["qc_pass"] = qc_pass
    raw["annotation_pass"] = annotation_pass
    candidates = raw.loc[qc_pass & annotation_pass].copy()
    candidates = candidates.sort_values(
        ["annotation_id", "Score", "qc_cv_percent", "Met ID"],
        ascending=[True, False, True, True],
    ).drop_duplicates("annotation_id", keep="first")
    candidates["feature"] = (
        candidates["metabolite_name_clean"]
        + "|"
        + candidates["annotation_id"]
        + "|"
        + candidates["ion_mode"]
    )
    intensities = candidates.set_index("feature")[sample_ids].T.astype(float)
    log_intensities = np.log2(intensities + 1.0)
    delta = log_intensities.loc[post_ids].set_axis(mapping["animal_id"], axis=0) - log_intensities.loc[pre_ids].set_axis(mapping["animal_id"], axis=0)
    delta.index.name = "animal_id"
    feature_qc = raw[
        [
            "Met ID",
            "Metabolite Name",
            "HMDB ID",
            "KEGG ID",
            "ion_mode",
            "Score",
            "qc_cv_percent",
            "qc_pass",
            "annotation_pass",
        ]
    ].copy()
    candidate_meta = candidates[
        [
            "feature",
            "Met ID",
            "metabolite_name_clean",
            "HMDB ID",
            "KEGG ID",
            "ion_mode",
            "Score",
            "qc_cv_percent",
        ]
    ].copy()
    candidate_meta["pre_mean_intensity"] = intensities.loc[pre_ids].mean(axis=0).reindex(candidate_meta["feature"]).values
    candidate_meta["post_mean_intensity"] = intensities.loc[post_ids].mean(axis=0).reindex(candidate_meta["feature"]).values
    return delta, feature_qc, candidate_meta, pd.DataFrame(normalization_rows)

def load_ko_delta(paths, mapping: pd.DataFrame, core) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(paths.ko)
    pre_ids = mapping["metagenome_pre_id"].tolist()
    post_ids = mapping["metagenome_post_id"].tolist()
    sample_ids = pre_ids + post_ids
    numeric = raw[sample_ids].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    totals = numeric.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("KO input contains a non-positive sample total")
    ppm = numeric.div(totals, axis=1) * 1_000_000.0
    annotated = raw["KOEntry"].notna()
    abundance = ppm.loc[annotated].set_axis(raw.loc[annotated, "KOEntry"]).T
    abundance = abundance.groupby(level=0, axis=1).sum()
    pre = abundance.loc[pre_ids].copy()
    post = abundance.loc[post_ids].copy()
    pre.index = mapping["animal_id"].to_numpy()
    post.index = mapping["animal_id"].to_numpy()
    delta = np.log2(post + PSEUDOCOUNT_PPM) - np.log2(pre + PSEUDOCOUNT_PPM)
    delta.index.name = "animal_id"
    stats = core.exact_signflip_pvalues(delta)
    stats = stats.rename(columns={"feature": "ko"})
    stats["pre_mean_ppm"] = stats["ko"].map(pre.mean(axis=0))
    stats["post_mean_ppm"] = stats["ko"].map(post.mean(axis=0))
    return abundance, delta, stats
