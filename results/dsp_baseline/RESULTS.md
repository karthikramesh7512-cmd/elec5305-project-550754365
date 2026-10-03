# First deterministic DSP baseline results

All 7,000 experimental images were scored. Ordinary and source-holdout protocols
use the unchanged existing manifests. Test sets were evaluated after all validation
choices were frozen. No model was trained and no test-driven retuning was performed.

All four windows achieved validation balanced accuracy 1.0 in every evaluation.
The predefined tie rule selected W=1 everywhere; these data do not establish
that W=1 is superior to the other windows. Threshold ties select the highest
threshold attaining the optimum, which here is the lowest validation-positive
score. Decisions use the full precision in the CSV, not the rounded values below.

| Evaluation | Test source | Original size | Threshold | TP | TN | FP | FN | Accuracy | Recall | Precision | F1 | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary | Mixed | Mixed | 4584.638764 | 517 | 533 | 0 | 0 | 1.000000 | 1 | 1 | 1 | 1 |
| Fold 1 | P41_01 | 2 | 6325.291403 | 0 | 593 | 0 | 547 | 0.520175 | 0 | 0 | 0 | 0.5 |
| Fold 2 | P41_02 | 3 | 14978.067378 | 0 | 593 | 0 | 543 | 0.522007 | 0 | 0 | 0 | 0.5 |
| Fold 3 | P41_03 | 26 | 13938.366294 | 621 | 593 | 0 | 0 | 1.000000 | 1 | 1 | 1 | 1 |
| Fold 4 | P41_04 | 6 | 10232.792205 | 558 | 593 | 0 | 0 | 1.000000 | 1 | 1 | 1 | 1 |
| Fold 5 | P41_05 | 17 | 7392.840322 | 574 | 593 | 0 | 0 | 1.000000 | 1 | 1 | 1 | 1 |
| Fold 6 | P41_06_notch | 6 | 4583.179837 | 599 | 593 | 0 | 0 | 1.000000 | 1 | 1 | 1 | 1 |

Specificity=1, false-positive rate=0, and ROC-AUC=1 in every evaluation.
Sensitivity is the recall column. Precision is defined as zero when no positive
predictions are made (folds 1 and 2); mathematically it is undefined in that case.

## Observed failures and limits

- Fold 1 validates on P41_02. Its threshold 6325.29 exceeds every held-out P41_01
  score (approximately 4583–4909), causing 547 false negatives.
- Fold 2 validates on P41_03. Its threshold 14978.07 exceeds every held-out P41_02
  score (approximately 6325–6365), causing 543 false negatives.
- ROC-AUC=1 is consistent with recall=0 at a selected threshold: the positives
  still rank above negatives, but the validation operating point fails to transfer.
- Every experimental no-flaw image has the same W=1 score, approximately
  3151.372023. This does not prove the images are identical, but shows no negative
  score diversity for this feature. Together with source-dependent score bands,
  it limits claims of generalization from perfect ordinary-split results.
- Balanced accuracy alone does not uniquely choose a threshold inside a perfectly
  separating validation gap. The prespecified highest-threshold tie policy matters
  for unseen sources. It was not changed after seeing the failures.

Full-precision metrics: `dsp_summary.csv`. Per-source recall and sizes:
`per_source_recall.csv`. All 1,090 test false negatives are listed in
`test_failure_cases.csv`; there are no false positives. Processing/score figures
are descriptive and were not used to revise the detector.

## Verification

Five unit tests passed (analytic envelope/DC/ROI invariance, valid spatial means,
threshold brute-force comparison, metrics/tied AUC, and test-data isolation).
`python verify_dsp_results.py` independently checked 49,000 prediction rows and
8,050 test predictions across seven evaluations, exact validation tuning IDs,
unchanged split hashes, and agreement between saved metrics and predictions.
The 8,050 test predictions belong to separate protocols: 1,050 ordinary plus
7,000 across six source-holdout folds.
