# Experimental evaluation manifests

Reproduce with `python create_evaluation_splits.py` from the project root.
Input: `data/metadata.csv`, filtered strictly to batches 013–019 (7,000 samples).
No simulated rows, signal processing, or model training are included.

- Ordinary split: 4,900 train / 1,050 validation / 1,050 test. Class quotas
  approximate 70/15/15 within one sample of proportional targets.
- Randomness: NumPy default_rng, seed 42 for ordinary allocation and seed 43
  for negative-group allocation. Input rows are sorted by batch and metadata row.
- Source order: P41_01, P41_02, P41_03, P41_04, P41_05, P41_06_notch.
  Fold k tests source k and validates the next source, wrapping at six.
- Six fixed negative groups contain 593 samples each. Rows are shuffled within
  each batch and assigned by a continuing round-robin. Contributions from any
  batch differ by at most one between groups. Fold k tests group k and validates
  the next group; the other four groups train.
- `source_label` is the recorded P41 label for positives; it is blank for negatives.
  `flaw_size` retains the original metadata value (zero for negatives).
- `sample_id` identifies batch plus one-based metadata row. `metadata_file` and
  `metadata_row` provide the original row reference. These identify metadata/image
  entries, not unique signal contents; no image-content deduplication was performed.
- `negative_group`, `test_source`, and `validation_source` document fold assignment.
  `negative_group` is blank for positives. Fold and group numbers are one-based.

The ordinary manifest has 7,000 rows. The holdout manifest has 42,000 rows:
each sample occurs once per fold, with one test, one validation, and four train
assignments across folds. The two manifests describe separate evaluation protocols.

The script checks class counts, exact coverage, metadata preservation, unique
sample IDs within each fold, source exclusion, negative-group balance and stability,
and rotation coverage before saving. It prints all partition class counts.
Source separation is at the P41-label level; it does not establish independence
of the shared experimental background or remove augmentation similarities.
