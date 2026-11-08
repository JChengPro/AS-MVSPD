"""Paired public-data preprocessing used for the manuscript."""

from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np

import pandas as pd

EPS = 1e-12

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def clr_transform(frame: pd.DataFrame, pseudocount: float | None = None) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    positive = values[values > 0]
    if pseudocount is None:
        pseudocount = float(np.min(positive) / 2.0) if positive.size else 0.5
    values = np.maximum(values, 0.0) + pseudocount
    values = values / values.sum(axis=1, keepdims=True)
    logged = np.log(values)
    clr = logged - logged.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=frame.index, columns=frame.columns)

def _pair_table(
    metadata: pd.DataFrame,
    sample_col: str,
    subject_col: str,
    order_col: str,
) -> pd.DataFrame:
    rows = []
    ordered = metadata.sort_values([subject_col, order_col, sample_col])
    for subject, group in ordered.groupby(subject_col, sort=True):
        if len(group) < 2:
            continue
        rows.append(
            {
                "subject_id": str(subject),
                "pre_sample": str(group.iloc[0][sample_col]),
                "post_sample": str(group.iloc[-1][sample_col]),
                "pre_order": str(group.iloc[0][order_col]),
                "post_order": str(group.iloc[-1][order_col]),
            }
        )
    return pd.DataFrame(rows)

def _paired_changes(
    microbiome: pd.DataFrame,
    metabolome: pd.DataFrame,
    pairs: pd.DataFrame,
    prevalence: float = 0.25,
    metabolite_missing: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    samples = pairs["pre_sample"].tolist() + pairs["post_sample"].tolist()
    missing_micro = sorted(set(samples) - set(microbiome.index.astype(str)))
    missing_metab = sorted(set(samples) - set(metabolome.index.astype(str)))
    if missing_micro or missing_metab:
        raise ValueError(
            f"Pair samples absent from matrices: microbiome={missing_micro}, metabolome={missing_metab}"
        )
    microbiome = microbiome.copy()
    metabolome = metabolome.copy()
    microbiome.index = microbiome.index.astype(str)
    metabolome.index = metabolome.index.astype(str)
    micro = microbiome.loc[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    micro = micro.loc[:, (micro > 0).mean(axis=0) >= prevalence]
    micro = micro.loc[:, micro.sum(axis=0) > 0]
    micro_clr = clr_transform(micro)

    metab = metabolome.loc[samples].apply(pd.to_numeric, errors="coerce")
    metab = metab.loc[:, metab.isna().mean(axis=0) <= metabolite_missing]
    medians = metab.median(axis=0)
    metab = metab.fillna(medians).loc[:, medians.notna()]
    metab = np.log2(np.maximum(metab, 0.0) + 1.0)

    subject_ids = pairs["subject_id"].astype(str).tolist()
    x_pre = micro_clr.loc[pairs["pre_sample"].astype(str)].set_axis(subject_ids)
    x_post = micro_clr.loc[pairs["post_sample"].astype(str)].set_axis(subject_ids)
    y_pre = metab.loc[pairs["pre_sample"].astype(str)].set_axis(subject_ids)
    y_post = metab.loc[pairs["post_sample"].astype(str)].set_axis(subject_ids)
    x_delta = x_post - x_pre
    y_delta = y_post - y_pre
    x_delta.index.name = "subject_id"
    y_delta.index.name = "subject_id"
    x_delta = x_delta.loc[:, x_delta.std(axis=0, ddof=1) > EPS]
    y_delta = y_delta.loc[:, y_delta.std(axis=0, ddof=1) > EPS]
    audit = {
        "subjects": len(pairs),
        "microbiome_features": x_delta.shape[1],
        "metabolome_features": y_delta.shape[1],
        "microbiome_prevalence_threshold": prevalence,
        "metabolite_missing_threshold": metabolite_missing,
    }
    return x_delta, y_delta, audit

def load_haddad(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    folder = root / "animal_exposure" / "haddad_osa"
    metadata_path = folder / "haddad_6weeks_metadata_matched.txt"
    microbiome_path = folder / "haddad_6weeks_deblur_otus_rare2k_matched_wtaxa.txt"
    metabolome_path = folder / "haddad_6weeks_allFeatures_pqn_matched_wIDs.txt"
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    metadata = metadata.copy()
    metadata["_order"] = pd.to_datetime(metadata["collection_timestamp"], errors="coerce")
    metadata = metadata.dropna(
        subset=["#SampleID", "host_subject_id", "exposure_type", "_order"]
    )
    pair_parts = []
    for exposure, group in metadata.groupby("exposure_type", sort=True):
        part = _pair_table(group, "#SampleID", "host_subject_id", "_order")
        part["exposure_type"] = exposure
        pair_parts.append(part)
    pairs = pd.concat(pair_parts, ignore_index=True)

    micro_raw = pd.read_csv(microbiome_path, sep="\t", low_memory=False)
    micro_id = micro_raw.columns[0]
    micro_annotations = [column for column in ["taxonomy"] if column in micro_raw]
    microbiome = micro_raw.drop(columns=micro_annotations).set_index(micro_id).T
    microbiome.index.name = "Sample"

    metab_raw = pd.read_csv(metabolome_path, sep="\t", low_memory=False)
    metab_id = metab_raw.columns[0]
    annotations = [
        column
        for column in ["obs_mz", "obs_rt", "level2_gnps", "level_1_identification"]
        if column in metab_raw
    ]
    metabolome = metab_raw.drop(columns=annotations).set_index(metab_id).T
    metabolome.index.name = "Sample"

    x_delta, y_delta, audit = _paired_changes(microbiome, metabolome, pairs)
    audit.update(
        {
            "dataset_id": "HADDAD_OSA_IHH",
            "exposure_counts": pairs["exposure_type"].value_counts().to_dict(),
            "microbiome_source_sha256": sha256(microbiome_path),
            "metabolome_source_sha256": sha256(metabolome_path),
            "metadata_source_sha256": sha256(metadata_path),
        }
    )
    return x_delta, y_delta, pairs, audit

def load_mars(root: Path):
    folder = root / 'borenstein_curated' / 'MARS_IBS_2020'
    metadata = pd.read_csv(folder / 'metadata.tsv', sep='\t', low_memory=False)
    metadata['_order'] = pd.to_numeric(metadata['Timepoint'], errors='coerce')
    metadata = metadata.dropna(subset=['Sample','Subject','_order'])
    pairs = _pair_table(metadata, 'Sample', 'Subject', '_order')
    micro = pd.read_csv(folder/'genera.tsv', sep='\t', low_memory=False).set_index('Sample')
    metab = pd.read_csv(folder/'mtb.tsv', sep='\t', low_memory=False).set_index('Sample')
    x, y, audit = _paired_changes(micro, metab, pairs)
    audit.update(dataset_id='MARS_IBS_2020')
    return x, y, pairs, audit


def load_he(root: Path):
    folder = root / 'borenstein_curated' / 'HE_INFANTS_MFGM_2019'
    meta = pd.read_csv(folder/'metadata.tsv', sep='\t')
    micro = pd.read_csv(folder/'genera.tsv', sep='\t').set_index('Sample')
    metab = pd.read_csv(folder/'mtb.tsv', sep='\t').set_index('Sample')
    if not (meta.Sample.is_unique and micro.index.is_unique and metab.index.is_unique):
        raise ValueError('Duplicate sample identifiers')
    meta['Age'] = pd.to_numeric(meta['Age'], errors='raise')
    selected = meta.loc[meta.Age.isin([2,12])].copy()
    if selected.duplicated(['Subject','Age']).any():
        raise ValueError('Duplicate infant-age records')
    selected = selected.loc[selected.Sample.isin(micro.index) & selected.Sample.isin(metab.index)]
    rows, excluded = [], []
    for subject, group in meta.groupby('Subject',sort=True):
        g = selected.loc[selected.Subject.eq(subject)]
        if set(g.Age) != {2,12}:
            excluded.append({'subject_id':str(subject),'reason':'Missing target age or cross-omics sample'})
            continue
        pre = g.loc[g.Age.eq(2)].iloc[0]; post = g.loc[g.Age.eq(12)].iloc[0]
        if pre['diet'] != post['diet']: raise ValueError('Inconsistent feeding group')
        rows.append(dict(subject_id=str(subject), pre_sample=pre.Sample, post_sample=post.Sample,
                         pre_order=2, post_order=12, diet=pre['diet']))
    pairs = pd.DataFrame(rows)
    for frame in (micro,metab):
        v=frame.loc[pairs.pre_sample.tolist()+pairs.post_sample.tolist()].apply(pd.to_numeric,errors='raise')
        if (v<0).any().any() or np.isinf(v.to_numpy()).any():raise ValueError('Invalid measurements')
    x,y,audit = _paired_changes(micro,metab,pairs)
    audit.update(dataset_id='HE_INFANTS_MFGM_2019',excluded_subjects=excluded)
    return x,y,pairs,audit

