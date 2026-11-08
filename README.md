# AS-MVSPD

Adaptive Sparse Multi-View Stable Paired Differential Detection for aligned,
paired high-dimensional multi-omics data. The joint test combines view-specific
sign-flip P-values using their maximum. Feature-level maxT inference is carried
out separately within each view.

## Installation

The reproduction environment was checked with Python 3.10.9. The pinned numerical
packages support Python 3.10–3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[reproduce,test]"
```

Alternatively, use `python -m pip install -r requirements.txt` and run the scripts
from this checkout. The software uses NumPy, pandas and SciPy. Figure generation
also uses Matplotlib and Pillow; rat Excel inputs use openpyxl.

## Run the included example

```bash
python examples/run_example.py
```

The example consists of **simulated paired-difference matrices**, with eight
subjects and six features per view. Expected joint P-value: `0.015625`; all 256
sign patterns are used. Expected machine-readable output is stored in
`examples/expected_output.json`. Actual results are written to `results/example/`.

To analyze your own matrices:

```bash
as-mvspd --view Microbiome=examples/microbiome_paired_changes.csv \
  --view Metabolome=examples/metabolome_paired_changes.csv \
  --k-grid 1 2 5 10 20 --seed 20260803 --output results/my_analysis
```

`python -m stablejoint` accepts the same arguments. Inputs are CSV matrices with
one paired individual per row, feature names in the header, and subject IDs in
the first column. Supply transformed **post-minus-pre differences**, not raw
abundances. All views must have identical subject order. See
[parameters and outputs](docs/PARAMETERS.md).

## Reproduce the manuscript

The public applications use HADDAD Air, HADDAD IHH, MARS and HE infants.

```bash
# Verify the implementation and run the bundled example.
python scripts/reproduce.py --stage check

# Download exact public-data versions, prepare pairs and run public analyses.
python scripts/reproduce.py --stage public

# Run the complete simulation configurations; this is computationally expensive.
python scripts/reproduce.py --stage simulations

# A reduced implementation check, with outputs isolated under results/quick/.
python scripts/reproduce.py --stage simulations --quick

# Render figures from the archived numerical source tables.
python scripts/build_figures.py --scope public
```

The default figure stage produces Figures 1–3 and S2–S4, S6–S10. Figure 4,
S1 and S5 require the rat figure-source tables. With those inputs:

```bash
python scripts/build_figures.py --scope all --rat-source PATH/TO/rat_figure_tables
```

Figures are exported as fixed-width PDF/SVG, RGB 1200-dpi TIFF and 300-dpi preview
PNG. Single-panel S8 is 86 mm wide; the other figures are 178 mm wide.
`--formats pdf png` selects lighter preview exports. These commands render
archived numeric results; they do not rerun simulations.

For recomputing statistics, preparing rat inputs and rebuilding all figure
source tables, see [the reproduction map](docs/REPRODUCIBILITY.md).

## Data and provenance

Public raw inputs are downloaded from pinned upstream commits and verified by
SHA-256. They are not bundled into the code release. The sample selection and
preprocessing functions are included in this repository. Rat data availability
must be specified separately; see [data documentation](docs/DATA.md).
The development workspaces are not needed to execute the code. Original source
file hashes and adaptation notes are recorded in
[the provenance manifest](docs/SOURCE_PROVENANCE.json).

## Release and citation

Version: `0.1.0` (local release preparation). The owner intends to publish the
repository under `JChengPro`; a remote repository and permanent archive have not
yet been created by this preparation. Before public release, the rights holders
must choose a software license and confirm the rat data-sharing arrangements.
No license grant or archive DOI is implied by this local preparation.

`CITATION.cff` contains the author names from the current manuscript. Add the
confirmed repository URL and version-specific archive DOI when available.
