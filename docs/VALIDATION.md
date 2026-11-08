# Local release validation

Environment: Python 3.10.9, NumPy 1.23.5, pandas 1.5.3, SciPy 1.10.0,
Matplotlib 3.7.0, openpyxl 3.0.10, Pillow 12.2.0, pytest 7.1.2.

Verified during preparation:

- The 14 original algorithm tests and two CLI regression tests passed.
- All nine independent comparator formula checks passed (maximum absolute
  error approximately 5.6e-16).
- Reimplemented public preprocessing reproduced all six archived matrices with
  identical subject and feature order; differences were at floating-point
  roundoff only.
- Extracted rat preprocessing reproduced the 8×745, 8×5475 and 8×1029 matrices
  to floating-point roundoff, without importing any former experiment project.
- Public joint P-values reproduced exactly: HADDAD Air 0.015625; HADDAD IHH
  0.0078125; MARS 0.03125; HE 0.00390625. The four Figure 3 comparator values
  per subset also matched the archived numeric inputs.
- All 143 public individual-deletion records, 360 view/joint feature-sampling
  records and 42 subgroup/reference records matched the figure source tables.
- The HE sensitivity summary reproduced the archived JSON exactly, including
  feeding groups, transformations and the 4096-pattern result.
- Rat main and KO-CLR sensitivity analyses were executed successfully; joint
  P-value 0.015625 in both analyses.
- The bundled synthetic example returned the documented joint P-value 0.015625.
- Reduced core and targeted-validation simulations completed. Full
  500-replicate simulations were not repeated in this release preparation.
- Main Figures 1–4 and Supplementary Figures S1–S10 were independently rendered
  from the included code and locally available source tables, without reading
  the manuscript Word files. The renderer reported no out-of-canvas text.
  S8 is 86 mm wide and the other 13 figures are 178 mm wide. Raster publication
  exports use RGB 1200-dpi TIFF; PNG previews are 300 dpi.
- An installable wheel was built with no dependency downloads. The installed
  package, bundled figure-style data and CLI example were tested outside the
  source checkout, using the already installed pinned dependencies. This was
  not a fresh dependency installation.

Cached inputs and verification logs are local development material and are not
included in the selected upload payload. Full-precision numeric output is
retained. Runtime depends on the machine and is not an equality check against
published elapsed seconds.

The final code ZIP was also extracted into a separate directory with no cached
study inputs or former experiment projects. All 16 tests, the example and all
11 public/simulation figures ran there successfully. All 11 fixed public-data
URLs were reachable; the three HADDAD files were downloaded and their SHA-256
checksums matched the original input bytes. Main and supplementary Word files
were unchanged. `docs/VALIDATION_REPORT.json` records the preparation results.
