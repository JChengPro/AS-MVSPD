# Parameters, inputs and outputs

## Input contract

Each CSV contains one paired subject per row and one feature per column. The
first column contains unique subject IDs. Entries are finite transformed
post-minus-pre differences. All views use the same subject order. Supply at
least two views and two individuals. Feature names must be unique within a view.
The public and rat preprocessing pipelines use their respective manuscript
transformations; see `preprocessing.py` and `local_preprocessing.py`.

## Command line

| Option | Default | Meaning |
| --- | --- | --- |
| `--view NAME=FILE.csv` | Required, repeated | Named paired-difference matrix; at least two |
| `--k-grid` | `1 2 5 10 20` | Candidate numbers of top features; values are clipped to the view's feature count and duplicates removed |
| `--seed` | `20260803` | Random seed for constructing a closed subgroup when full enumeration exceeds the budget |
| `--max-random` | `4095` | Non-identity sign-pattern budget; includes the identity in the resulting group |
| `--output` | `results/example` | Directory for results |

`generate_signs` enumerates the full group if `2**n <= max_random + 1`.
Otherwise it enumerates a closed GF(2) subgroup of size
`2**floor(log2(max_random + 1))`, capped by the full group size. This is not a
set of arbitrary independently sampled sign patterns. The identity is first.
The manuscript uses all 256 patterns for each n=8 application and closed
256-element subgroups for MARS and HE. `--max-random 255` requests that budget.

## Python API

```python
import pandas as pd
from stablejoint import analyze_multiview, generate_signs

blocks = {
    "Microbiome": pd.read_csv("examples/microbiome_paired_changes.csv", index_col=0),
    "Metabolome": pd.read_csv("examples/metabolome_paired_changes.csv", index_col=0),
}
signs, full_group = generate_signs(8, max_random=4095, seed=20260803)
result = analyze_multiview(blocks, signs=signs, k_grid=(1, 2, 5, 10, 20))
print(result.joint_pvalue)
```

`analyze_multiview` accepts `blocks`, `signs=None`, `k_grid=(1,2,5,10,20)`,
`influence_constraint=False`, `max_random=4095`, and `seed=20260803`.
If `signs` is supplied, it overrides sign generation from `max_random` and `seed`.
The optional `influence_constraint=True` selects the LOO ablation variant;
the primary AS-MVSPD implementation uses `False`.

## Output fields

The CLI creates `summary.json` and `view_1_feature_results.csv`, etc. Ordinal
filenames follow the order of `--view` arguments. The JSON reports view P-values,
their maximum (`joint_pvalue`), selected k values and the sign-group settings.
Feature tables include self-normalized scores, paired effect summaries,
unadjusted feature P-values and within-view `maxT_fwer` values. The API also
returns aggregation-scale and permutation tables. Full-precision numeric values
are retained in output files; presentation rounding is applied in the manuscript.

Simulation settings and random seeds are fixed in `config/experiment.json`.
The fixed-scale experiment records its additional seeds and settings in its
`design.json`. Subgroup seed conventions for Figure 3 and the HE checks are
listed in `docs/REPRODUCIBILITY.md`.
