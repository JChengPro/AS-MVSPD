# Data inputs and access

## Included with the code

- `examples/`: small simulated paired-difference matrices and expected output.
- `figures/source_data/`: numerical tables for public applications and simulation
  panels, plus aggregate method-comparison results.
- `reference/`: HE primary and sensitivity summaries needed to redraw Figure S10.

The included example is synthetic. It is not a substitute for the study inputs.

## Public studies

Run `python scripts/download_public_data.py`, then
`python scripts/01_prepare_inputs.py --scope public`. Use `--verify-only` on the
downloader to check an existing cache without network access.

| Study | Upstream repository | Commit |
| --- | --- | --- |
| HADDAD Air/IHH (Tripathi et al., 2018) | https://github.com/knightlab-analyses/haddad_osa | `44320d64845ce3d805821f1c22c7b090b75c0a11` |
| MARS (Mars et al., 2020) | https://github.com/borenstein-lab/microbiome-metabolome-curated-data | `89a519d8c832008fbc6e650453e83e2f04858d02` |
| HE infants (He et al., 2019) | Same curated resource, `HE_INFANTS_MFGM_2019` | `89a519d8c832008fbc6e650453e83e2f04858d02` |

Exact filenames, source URLs and checksums are in `config/public_sources.json`.
Upstream data remain governed by their own access and reuse terms. The code
release does not grant a new license over third-party data.

HADDAD and MARS use the first and last available samples per individual.
HADDAD preprocessing is performed on the matched study matrices before splitting
Air and IHH, matching the archived experiment. HE uses complete 2- and 12-month
pairs across the original feeding groups. Microbial prevalence, metabolite
missingness, CLR and log transformations are defined in `stablejoint/preprocessing.py`.
Expected retained dimensions:

| Subset | Paired individuals | Microbial features | Metabolite features |
| --- | ---: | ---: | ---: |
| HADDAD Air | 8 | 93 | 1707 |
| HADDAD IHH | 8 | 93 | 1707 |
| MARS | 70 | 701 | 26 |
| HE infants | 57 | 33 | 119 |

## Rat study

Rat data and individual rat figure-source tables are excluded from the prepared
code-upload inventory while the authors confirm the actual sharing arrangements.
A local verification cache may exist in the working directory; it is excluded by
`.gitignore` and by `scripts/prepare_release.py`.

For preprocessing, supply the original species workbook, KO workbook, positive/
negative metabolite workbook, paired-sample mapping CSV, and KO annotation table.
Copy `config/local_inputs.example.json` outside the release directory, replace
its example paths, then run:

```bash
python scripts/reproduce.py --stage rats --inputs /path/to/local_inputs.json
```

Paths inside the JSON may be absolute or relative to the JSON file. The sample
mapping requires `animal_id`, `species`, `included_in_cross_omics_analysis`,
`metagenome_pre_id`, `metagenome_post_id`, `metabolome_pre_id`, and
`metabolome_post_id`. QC figures additionally use `metagenome_pair_present`,
`metabolome_pair_present` and `cross_omics_complete`.

Prepared matrices are `data/local/species_paired_changes.csv`,
`ko_paired_changes.csv`, and `metabolite_paired_changes.csv`. Annotation and QC
outputs are generated alongside them. Expected shapes are 8×745, 8×5475 and
8×1029. The main analysis additionally reads the KO and metabolite annotation
metadata to label features.

`--rat-source` for the figure builder accepts generated `local_*.csv` and
`qc_*.csv` plotting tables. These contain individual-level information and need
the same deliberate data-sharing decision as the prepared study matrices.

The final manuscript must give the verified data access arrangement and any
accession or archive links. This preparation does not invent a restriction,
public accession or an upon-request access route.
