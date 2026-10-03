# Methodology and packaging record

The current scientific narrative is in the root README. Original evidence remains
at its original data/results paths to preserve provenance hashes:

- [Source audit](../../data/source_flaw_audit_notes.md)
- [Split definitions](../../data/splits/README.md)
- [Data inspection](../../results/data_inspection/inspection_report.md)
- [DSP procedure](../../results/dsp_baseline/README.md)
- [DSP results](../../results/dsp_baseline/RESULTS.md)
- [DSP diagnostics](../../results/dsp_baseline/dsp_diagnostics.md)
- [SVM results and feature definitions](../../results/svm/svm_report.md)

These historical reports are preserved verbatim. Their root-level Python commands
predate packaging: use `python src/<script>.py`, or `python tests/verify_dsp_results.py`
with `src` on PYTHONPATH, as described in the root README. Earlier DSP reports only
observed identical negative scores; the later raw-image hash audit established
that all 3,558 experimental negative rows contain exactly the same image.

Packaging changed source root resolution from `.parent` to `.parents[1]`, updated
source-file provenance paths, and normalised historical Windows path separators
when tests read provenance keys. The historical DSP source hash is checked after
undoing only its root-resolution change. Nine unit/provenance tests and the
independent DSP saved-result verifier passed after relocation.
No scientific computations, splits, model outputs, or data/results files changed.
The excluded IDE launcher was redirected to `src/audit_dataset.py`.
`packaging_preservation.json` records SHA-256 hashes of every original data/results
file and the relocated implementation files. Saved SVM models are included because
the existing provenance tests check their training-only fits and predictions.


