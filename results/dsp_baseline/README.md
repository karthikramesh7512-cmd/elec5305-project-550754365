# Deterministic DSP baseline

Run `python dsp_baseline.py` from the project root. Run checks with
`python -m unittest -v test_dsp_baseline.py`.
After a completed run, use `python verify_dsp_results.py` to check saved artifacts.
See `RESULTS.md` for the completed run and failure analysis.

Only XZ batches 013–019 and the existing ordinary/source-holdout manifests are
used. The detector receives an image array only. Neither metadata depth nor
location is loaded; source labels and sizes are attached only for reporting.

## Fixed pipeline

1. Decode each image as little-endian int16, C-order shape (480, 7168).
2. Take exactly `image[:, 1100:3100]` for every sample, convert to float64.
3. Subtract each A-scan's own mean over its 2,000 ROI samples.
4. Compute `abs(scipy.signal.hilbert(centered_roi, axis=1))`.
5. Take the maximum envelope amplitude along the time axis for each scan line.
6. Apply a spatial moving average of width 1, 3, 5 or 7. Use `valid` convolution:
   only complete windows, no artificial spatial edge padding.
7. Take the maximum spatial response as the continuous image score.

Hilbert is FFT-based, with no extra temporal padding or edge trimming in this
baseline. Its magnitude is the amplitude envelope; see the
[SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html).
No filtering, scaling learned across images, background subtraction, or detector
postprocessing is added. Per-A-scan DC subtraction uses only that input image,
so no population preprocessing statistics need to be fitted on training data.

## Selection and test isolation

Each scheme/fold selects W and threshold only on its own validation partition.
Candidate thresholds are the distinct validation scores plus a value just above
the largest score (allowing an all-negative classifier). Predict flaw when
`score >= threshold`. Optimize balanced accuracy. Exact ties choose the highest
threshold within W, then the smallest W. ROC-AUC uses continuous scores and
average ranks for ties; undefined precision when no positives are predicted is
reported as zero.

All seven selections are saved before computing any test metrics. Candidate
scores can be computed once for all samples because scoring has no fitted
parameters and uses no labels. No test scores or labels enter tuning.
Within source-holdout evaluation a sample may legitimately play different roles
in different folds; test isolation is enforced separately for each fold.

## Artifacts

- `candidate_scores.csv`: four unthresholded scores for all 7,000 samples.
- `validation_candidates.csv`: best threshold and validation balanced accuracy
  for each of the four W values in every evaluation.
- `selected_parameters.csv`: frozen selections.
- `tuning_sample_ids.csv`: exact validation IDs used in each selection.
- `per_sample_results.csv`: score, prediction, truth and provenance for every
  partition; ordinary fold is 0, source-holdout folds are 1–6.
- `dsp_summary.csv`: ordinary and six source-holdout test metrics.
- `per_source_recall.csv`: test recall by source and original size.
- `test_failure_cases.csv`: all test false positives/false negatives for inspection.
- `checks.json`: split SHA-256 digests, invariants, settings and software versions.
- PNGs: two processing examples selected from ordinary training without using
  detector scores; separate train/validation and test distributions for each
  evaluation; ordinary test confusion matrix; held-out-source recall.

The score-distribution plots combine train and validation only in the development
panel; test data are always plotted separately. Each plot uses the corresponding
evaluation's validation-selected W and threshold.

Input split files are never opened for writing and their hashes are compared
before/after. Coverage checks enforce exactly one prediction for each expected
test sample within its evaluation. Unit tests also perturb test labels/scores,
inject irrelevant metadata fields, test out-of-ROI and DC invariance, compare
threshold search with brute force, and verify ROC-AUC handling of tied scores.

This is a single prespecified baseline evaluation. Test errors are descriptive;
they are not used to revise W, thresholds, or preprocessing. No SVM or CNN is
implemented. Shared experimental backgrounds can make ordinary image-level
performance optimistic; source-holdout evaluation addresses the P41 source labels,
not independent specimens or backgrounds.
