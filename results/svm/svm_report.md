# Engineered-feature SVM: completed experiment

## Scope and main result

Only experimental batches 013–019 and the existing ordinary/six source-holdout
manifests were used. Feature extraction and saved model fits were reused for this
finalization; no real-data model was refitted and no test-set retuning was performed.
DSP Baseline V1 and all split files remain unchanged. No CNN is implemented.

**All 3,558 no-flaw experimental images are exact duplicates.** Therefore,
specificity cannot be interpreted as generalisation to novel negative examples.
Ordinary image-level performance is mainly a reference result. Source-flaw holdout
is the main generalisation result, limited to these six sources on a shared
experimental background; it does not establish new-specimen generalisation.

P41_01 has **540 TP / 7 FN**, recall **540/547 = 98.7202925%**. P41_02 and all
other held-out sources have 100% recall. These improve on DSP V1 for P41_01 and
P41_02; no held-out source worsens. There are no false positives on the duplicated
negative background. The seven P41_01 errors cover 5
distinct raw-image hashes, so even the false-negative rows are not all independent.

## Selected models and validation comparison

| scheme | fold | kernel | C | gamma | validation_balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| ordinary | 0 | linear | 0.01 | not applicable | 1 |
| source_holdout | 1 | linear | 0.01 | not applicable | 1 |
| source_holdout | 2 | linear | 0.01 | not applicable | 1 |
| source_holdout | 3 | linear | 0.01 | not applicable | 1 |
| source_holdout | 4 | linear | 0.01 | not applicable | 1 |
| source_holdout | 5 | linear | 0.01 | not applicable | 1 |
| source_holdout | 6 | rbf | 0.1 | 1 | 1 |

Candidates: linear C in [0.01,0.1,1,10,100]; RBF C in [0.1,1,10,100] with
gamma in [scale,0.001,0.01,0.1,1]. Validation balanced accuracy alone selects.
Ties follow fixed grid order: linear first, ascending C, then listed gamma order.
Thus a winning linear model need not be uniquely best. The RBF fold has no
direct linear coefficient interpretation. Gamma is unused for linear models.

Best validation balanced accuracy per kernel:

| scheme | fold | linear | rbf |
| --- | --- | --- | --- |
| ordinary | 0 | 1 | 1 |
| source_holdout | 1 | 1 | 1 |
| source_holdout | 2 | 1 | 1 |
| source_holdout | 3 | 1 | 1 |
| source_holdout | 4 | 1 | 1 |
| source_holdout | 5 | 1 | 1 |
| source_holdout | 6 | 0.994516 | 1 |

## Complete test metrics

All rates below are fractions. Sensitivity equals recall. PR-AUC is trapezoidal
area under the precision-recall curve of SVC decision scores; average precision
is saved separately and is also 1.0 in each evaluation. Scores are uncalibrated
decision values, not probabilities. SVC's original decision rule was retained.

| test_source | TP | TN | FP | FN | accuracy | recall | specificity | precision | F1 | false_positive_rate | balanced_accuracy | ROC_AUC | PR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary | 517 | 533 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| P41_01 | 540 | 593 | 0 | 7 | 0.99386 | 0.987203 | 1 | 1 | 0.99356 | 0 | 0.993601 | 1 | 1 |
| P41_02 | 543 | 593 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| P41_03 | 621 | 593 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| P41_04 | 558 | 593 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| P41_05 | 574 | 593 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| P41_06_notch | 599 | 593 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |

## DSP V1 versus SVM

| evaluation | held_out_source | flaw_size | DSP_recall | SVM_recall | DSP_balanced_accuracy | SVM_balanced_accuracy | DSP_ROC_AUC | SVM_ROC_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary | mixed | 2.0;3.0;6.0;17.0;26.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| Source holdout 1 | P41_01 | 2.0 | 0 | 0.987203 | 0.5 | 0.993601 | 1 | 1 |
| Source holdout 2 | P41_02 | 3.0 | 0 | 1 | 0.5 | 1 | 1 | 1 |
| Source holdout 3 | P41_03 | 26.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| Source holdout 4 | P41_04 | 6.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| Source holdout 5 | P41_05 | 17.0 | 1 | 1 | 1 | 1 | 1 | 1 |
| Source holdout 6 | P41_06_notch | 6.0 | 1 | 1 | 1 | 1 | 1 | 1 |

Per-source recall differences (SVM minus DSP; fractions):

| source_flaw | flaw_size | svm_recall | dsp_recall | recall_difference |
| --- | --- | --- | --- | --- |
| P41_01 | 2 | 0.987203 | 0 | 0.987203 |
| P41_02 | 3 | 1 | 0 | 1 |
| P41_03 | 26 | 1 | 1 | 0 |
| P41_04 | 6 | 1 | 1 | 0 |
| P41_05 | 17 | 1 | 1 | 0 |
| P41_06_notch | 6 | 1 | 1 | 0 |

## Feature definitions and audit

Fixed ROI `image[:,1100:3100]`, float64, independent DC removal per A-scan,
Hilbert envelope along axis 1. Feature extraction receives only image pixels.
Labels, source, size, depth, location, IDs and hashes are excluded from model X;
X uses an explicit allowlist of exactly 12 feature columns.

| Feature | Definition |
| --- | --- |
| max_envelope | Maximum Hilbert-envelope amplitude in fixed ROI |
| p95_envelope | 95th percentile of all ROI envelope values (linear interpolation) |
| mean_envelope | Mean of all ROI envelope values |
| rf_rms | RMS of DC-removed RF ROI |
| rf_energy | Sum of squared DC-removed RF samples; 960000 * RMS squared |
| envelope_energy | Sum of squared envelope samples |
| crest_factor | Maximum absolute DC-removed RF / RF RMS; zero for zero signal |
| strong_peak_count | Spatial line-response local peaks: height >= 0.5 max, prominence >= 0.1 max, distance >= 3 scan indices; endpoints eligible |
| spatial_persistence | Longest contiguous scan run with line response >= 0.5 max |
| strongest_response_width | Interpolated half-maximum width of run containing global spatial maximum; clipped at image edge |
| line_peak_std | Population SD over 480 per-A-scan maximum envelope values |
| line_peak_mean | Mean over 480 per-A-scan maximum envelope values |

Population SD uses ddof=0. Near-constant means one value occupies >=99% of rows,
or SD <= 1e-6 * max(abs(mean),1), including constants. No feature meets those
criteria globally. All features are constant within the duplicated no-flaw class.
No features were removed. Full-data audit/correlation statistics are descriptive
only: they did not affect the feature list, scaling, model grid or selection.

| feature | mean | std | minimum | maximum |
| --- | --- | --- | --- | --- |
| max_envelope | 6375.25 | 4246.91 | 3151.37 | 15030 |
| p95_envelope | 1367.95 | 75.2714 | 1317.22 | 1571.2 |
| mean_envelope | 545.314 | 20.8696 | 531.693 | 601.13 |
| rf_rms | 510.25 | 57.5255 | 474.32 | 662.502 |
| rf_energy | 2.53118e+11 | 6.1558e+10 | 2.15981e+11 | 4.21353e+11 |
| envelope_energy | 5.06236e+11 | 1.23116e+11 | 4.31961e+11 | 8.42705e+11 |
| crest_factor | 11.7165 | 6.40528 | 6.59972 | 24.7622 |
| strong_peak_count | 22.4761 | 17.0497 | 1 | 38 |
| spatial_persistence | 50.6374 | 38.1236 | 8 | 88 |
| strongest_response_width | 50.2227 | 38.6473 | 6.65127 | 88.0712 |
| line_peak_std | 830.716 | 538.779 | 481.764 | 2139.44 |
| line_peak_mean | 2204.6 | 184.009 | 2082.93 | 2709.72 |

Per-feature correlation partners are in `feature_audit.csv`; all pairs with
absolute Pearson r >=0.95 are in `redundant_features.csv`. Strong redundancy:
RF energy vs envelope energy r≈1; RMS vs RF energy r≈0.999016; persistence vs
strongest-response width r≈0.999358; envelope p95 vs mean r≈0.996867.
RF energy is exactly 960000 * RF RMS squared. Envelope energy is nearly twice
RF energy for these centered RF traces. Keeping correlated features changes the
geometry seen by the SVM; coefficients are consequently not independent effects.

## Standardised linear coefficients

The signed weights below multiply training-standardised features; positive values
increase the flaw-class decision value, with other standardised inputs fixed.
They are **not causal feature importances**. Coefficient magnitudes and ranks
for every selected linear model are in `selected_linear_coefficients.csv`.

| feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Ordinary |
| --- | --- | --- | --- | --- | --- | --- |
| max_envelope | 0.116848 | 0.122553 | 0.170489 | 0.134904 | 0.11697 | 0.117633 |
| p95_envelope | 0.0630742 | 0.259919 | 0.24506 | 0.213816 | 0.220996 | 0.20927 |
| mean_envelope | 0.055266 | 0.181447 | 0.17625 | 0.149455 | 0.152924 | 0.146521 |
| rf_rms | 0.0490978 | 0.111838 | 0.117022 | 0.0822309 | 0.0833138 | 0.0833354 |
| rf_energy | 0.0427581 | 0.102618 | 0.106641 | 0.0695791 | 0.07115 | 0.0719914 |
| envelope_energy | 0.0427581 | 0.102617 | 0.106641 | 0.0695791 | 0.07115 | 0.0719914 |
| crest_factor | 0.14956 | 0.127321 | 0.177012 | 0.16531 | 0.134589 | 0.133486 |
| strong_peak_count | -0.269885 | -0.208509 | -0.188113 | -0.238603 | -0.253911 | -0.222837 |
| spatial_persistence | -0.255439 | -0.435789 | -0.390594 | -0.495185 | -0.514346 | -0.491512 |
| strongest_response_width | -0.250226 | -0.433173 | -0.389967 | -0.485131 | -0.502089 | -0.484246 |
| line_peak_std | 0.0697383 | 0.046981 | 0.0662643 | 0.0403453 | 0.0373583 | 0.0379854 |
| line_peak_mean | 0.0816091 | 0.104351 | 0.123624 | 0.0804027 | 0.0787359 | 0.0746553 |

The largest absolute weights are usually persistence and strongest-response width,
both negative. Fold P41_01 instead gives its largest magnitude to strong-peak
count (-0.269885), followed by persistence (-0.255439) and width (-0.250226).
In the ordinary model the three largest are persistence (-0.491512), width
(-0.484246), and strong-peak count (-0.222837).

The repeated negative background has persistence 88 scan samples, strongest width
88.0712 and 38 strong peaks. Positive median persistence is 8–15 and peak counts
1–27. Because the threshold is relative to each image's maximum, stronger local
flaw echoes can concentrate the above-half-maximum response. This is a supported
feature observation, not physical crack-width measurement or a causal claim.
RBF fold P41_06_notch has no direct linear coefficients. The older
`linear_feature_coefficients.csv` additionally records its best *unselected*
linear candidate; that candidate must not be interpreted as the selected RBF.

## Failure cases and warnings

| sample_id | source_label | decision_score | predicted_label |
| --- | --- | --- | --- |
| batch_013:245 | P41_01 | -0.127933 | 0 |
| batch_015:244 | P41_01 | -0.125576 | 0 |
| batch_015:873 | P41_01 | -0.125576 | 0 |
| batch_017:536 | P41_01 | -0.142102 | 0 |
| batch_017:876 | P41_01 | -0.00058573 | 0 |
| batch_018:839 | P41_01 | -0.141229 | 0 |
| batch_019:642 | P41_01 | -0.142102 | 0 |

All seven missed P41_01 positives have negative decision values. One is near the
boundary (-0.000586); the others are approximately -0.126 to -0.142. No adjustment
was made after seeing them. ROC-AUC and PR-AUC remain 1.0 because positive scores
still rank above negative scores even when the zero decision boundary misses some.
Ranking performance and recall at an operating point answer different questions.

The negative duplication makes specificity an observation about one reused
background, not a performance estimate for diverse negative acquisitions.
Ordinary perfect performance must be read as a reference result, with duplicated
content and related augmented positives across partitions. The six held-out-source
results are more informative but retain the shared-background limitation.

## Figures

- `dsp_vs_svm_balanced_accuracy.png` and `dsp_vs_svm_recall.png`.
- `ordinary_confusion_matrix.png`.
- `source_holdout_confusion_matrices.png` plus six `confusion_fold_*.png` files.
- `feature_correlation.png`, `feature_distributions.png` (ordinary train/validation).
- `pca_diagnostic.png`: diagnostic only. PCA and its scaler fit ordinary training
  rows only; all partitions are transformed for display. PCA is not a classifier input.

## Verification and reproducibility

Run `python -m unittest discover -v` for feature/DSP unit and SVM provenance tests.
`test_results.txt` records the final test execution. Checks cover:

- Scaler fitted on training rows only: saved n_samples_seen, means and variances
  agree with training data.
- SVC fitted on training only: fit-call tests, logged IDs, saved fit shape and
  support vectors matching training-scaled rows.
- Validation labels used for hyperparameter selection, never scaler/SVC fitting.
- No test IDs in fitting or selection; test labels never enter model selection.
- Held-out sources absent from training; exactly one prediction per expected test
  sample within each protocol/fold (8,050 total test predictions across protocols).
- Frozen DSP outputs/splits retain SHA-256 hashes; saved SVM models, features,
  summary and predictions remain unchanged during finalization.

Pipeline: `StandardScaler()` then `SVC()`. No train+validation refit, threshold
retuning, class weighting, oversampling, feature removal, or PCA classification.
The pre-existing duplicate negatives were retained as required by the fixed splits.
Dependencies are installed in project-local `.python_deps`; sklearn version is
1.9.1. `extract_features.py` and `svm_baseline.py` reproduce the
original extraction/evaluation; `finalize_svm_report.py` updates reporting only.
