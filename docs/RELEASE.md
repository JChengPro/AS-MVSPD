# Release preparation and journal requirements

The locally saved Bioinformatics author guidelines require a self-contained,
easy-to-use software implementation with test data, acquisition/installation/
execution instructions, and descriptions of optional parameters. At submission,
software and source must be freely accessible to non-commercial users without
request at stable URLs. The submitted software version and test data must also
be archived at a dedicated repository, with its URL in the abstract availability
section. Instructions to reproduce manuscript results and figures are required.
An open-source license and Code Ocean deposition are encouraged.

This summary is based on the author-supplied saved official HTML. It is not a
claim that the live author-guidelines page was rechecked during this preparation.
Official entry: https://academic.oup.com/bioinformatics/pages/author-guidelines

## Items requiring an author decision or an external action

1. Confirm the rights holder(s) and select a license. MIT is a reasonable simple
   option for an author-owned Python method; this preparation does not grant it
   on behalf of the authors. A final LICENSE must contain the confirmed terms.
2. Confirm the actual public or reviewer access arrangements for rat study inputs
   and individual-level figure tables, and assess third-party data redistribution
   terms before adding such material to the archive.
3. Create the intended GitHub repository (suggested name `AS-MVSPD`), inspect the
   staged file list, then upload the chosen release. No GitHub write is performed
   by this preparation.
4. Tag the submitted version, e.g. `v0.1.0`, and archive that exact version **with
   the example/test data** in Zenodo or another supported dedicated repository.
   GitHub releases alone are not treated here as fulfilling that archive clause.
5. Add the actual repository and version-specific archive links to CITATION.cff,
   README and the manuscript's availability statements. No DOI is preassigned.

## Export only the reviewed files

```bash
python scripts/prepare_release.py
```

This creates an explicit upload file list and a local ZIP outside the repository.
Only package source, selected scripts, configuration, examples, tests, reference
summary tables and documentation are selected. The exporter excludes local rat
inputs, public raw caches, generated figures, `.private`, results, verification
logs, Word documents and spreadsheet files. It does not upload anything or
remove the original research history.

`release_readiness.json` reports pending decisions honestly. A preparation ZIP
can be generated before they are settled; its existence does not mean the
submission is ready. Add a confirmed LICENSE before making the public release.
