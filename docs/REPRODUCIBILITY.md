# Reproduction map

There are two supported paths: redraw from archived numeric tables, or recompute
statistics from inputs. The figure builder never reruns a statistical experiment
implicitly. Execute commands from the repository root.

## Result and figure map

| Manuscript content | Computation | Drawing |
| --- | --- | --- |
| Figure 1 | Method schematic; no numeric experiment | `scripts/build_figures.py` |
| Figure 2; S2 | `02_run_simulation.py`, `09_run_method_validation.py`, `11_run_structured_correlation_validation.py`, `13_run_asymmetry_gradient.py` | Same builder |
| Figure 3; S4; Table 1 | `03_run_public_validation.py --sensitivity` | Same builder |
| Figure 4; S1; S5 | `01_prepare_inputs.py --scope local`, `04_run_local_application.py`, `06_prepare_figure_data.py --section local` | Same builder, `--scope all` |
| S3 A–C | `06_prepare_figure_data.py --section extended --refresh-expensive` | Same builder |
| S3 D–E | `03_run_public_validation.py --sensitivity` | Same builder |
| S3 F | `16_benchmark_runtime.py` | Same builder |
| S6 | `11_run_structured_correlation_validation.py` | Same builder |
| S7 | `13_run_asymmetry_gradient.py` | Same builder |
| S8 | `09_run_method_validation.py` | Same builder |
| S9 | `18_fixed_scale_ablation.py` | Same builder |
| S10; HE supplementary results | `run_he_validation.py`, `run_he_sensitivity.py` | Same builder |
| KO CLR sensitivity | `15_run_ko_clr_sensitivity.py` | Numeric outputs under `results/local_application/ko_clr_sensitivity/` |
| Sign groups | `17_audit_sign_groups.py` | JSON with seeds, dimensions and hashes |

All simulation scripts above are under `scripts/`. Principal configurations use
500 replicates; contamination validation uses 250. Full configurations are
retained. `--quick` runs reduced checks only and must not be used for manuscript
estimates. The convenience `--stage simulations --quick` runs the core two-view
and targeted validation checks, and writes to `results/quick/`.

## Redraw archived results

```bash
python scripts/build_figures.py --scope public
python scripts/build_figures.py --scope all --rat-source /path/to/rat_figure_tables
```

The second command requires the rat tables described in `DATA.md`. The local
preparation directory contains a private verification copy; the exported code
inventory does not. Legends are extracted from the current manuscript and stored
in `figures/legends.json`. No manuscript DOCX is needed to render figures.

## Recompute all statistics and then draw

```bash
python scripts/reproduce.py --stage public
python scripts/reproduce.py --stage simulations
python scripts/reproduce.py --stage rats --inputs /path/to/local_inputs.json
python scripts/09_run_method_validation.py
python scripts/16_benchmark_runtime.py
python scripts/reproduce.py --stage tables
python scripts/build_figures.py --scope all \
  --source-data results/recomputed_figure_data \
  --rat-source results/recomputed_figure_data
```

The second invocation of method validation supplies real-data comparisons after
all input matrices exist; it also reruns its simulation sections. To avoid
that extra computation, `scripts/recompute_comparisons.py` recomputes just the
real-data comparison table. `tables` recomputes extended simulations, public
sensitivity tables and local plot tables, then assembles the inputs. It fails
when a required table is absent rather than silently substituting archived
results. Runtime measurements depend on CPU, BLAS and system load and are not
expected to match published wall times exactly.

Individual stages can be run using `06_prepare_figure_data.py --section` with
`simulation`, `extended`, `validation`, `public`, or `local`.

## Randomization conventions

Master seed: `20260803`. Public primary subgroup seed: master + sample size.
Figure 3 individual-deletion subgroup seed: master + 7000 + remaining sample size.
The public feature-sampling stream starts at master + 7000 and processes Air,
IHH, then MARS. HE feature sampling uses master + 40000. MARS and HE Figure 3/
S3/S4 additional subgroup seeds are master + 7100 through master + 7119.
The separate HE checks use primary seed + 1 through + 20; their additional
4096-element subgroup uses the HE primary seed. The two sensitivity seed sets
are documented separately to preserve the original calculations.

`17_audit_sign_groups.py` records float64 row-major matrix hashes. The fixed-scale
experiment additionally records an int8 matrix hash in its `design.json`; these
different hash encodings should not be compared as if they were the same.

## Software and hardware record

The numerical environment is pinned in `requirements.txt`; the verified Python
version and test outcomes are recorded in `docs/VALIDATION.md`. Thread settings
and wall-clock summaries are recorded by the runtime script. This release
preparation has not rerun every 500-replicate simulation. The corresponding
source code, fixed configuration and archived numeric summaries are included.

## Supplementary table map

The numerical contents of the submitted workbook correspond to the following
outputs. Workbook styling is separate from statistical reproduction.

| Supplementary table | Numeric source / generating code |
| --- | --- |
| S1 | `simulation_performance_with_ci.csv`; `02_run_simulation.py` then `06_prepare_figure_data.py --section simulation` |
| S2 | `simulation_feature_recovery_with_se.csv`; same commands |
| S3 | `simulation_extended_with_ci.csv`; `06_prepare_figure_data.py --section extended --refresh-expensive` |
| S4 | Public tests, method comparisons and resampling; `03_run_public_validation.py --sensitivity`, `recompute_comparisons.py` |
| S5 | Rat feature result CSVs from `04_run_local_application.py` and `local_all_feature_ranks.csv` from the local figure-table stage |
| S6 | Asymmetry joint/FWER outputs from `13_run_asymmetry_gradient.py` |
| S7 | `analysis_summary.csv`, `feature_stability_summary.csv`, `ko_clr_feature_results.csv` from `15_run_ko_clr_sensitivity.py` |
| S8 | `results/reproducibility/sign_group_audit.json` from `17_audit_sign_groups.py` |
| S9 | `results/publication_figures/audit/Figure2_data_audit.csv` from `19_audit_figure2_data.py` |
| S10 | Fixed-scale summary, paired comparisons and replicate-level outputs from `18_fixed_scale_ablation.py` |
| S11 | Primary, seed sensitivity and further-audit CSV/JSON files from `run_he_validation.py` and `run_he_sensitivity.py` |

For S9, `python scripts/19_audit_figure2_data.py` audits archived numeric tables.
Use `--source-data results/recomputed_figure_data` to audit recomputed results.
It checks panel point counts and 500-replicate counts, without loading Word.
