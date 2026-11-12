# Final CNN results

Across the six held-out sources, CNN macro recall is 85.07% (seed SD 1.02 percentage points), compared with 66.67% for DSP and 99.79% for the SVM. CNN macro balanced accuracy is 92.54% (seed SD 0.51 points), versus 83.33% for DSP and 99.89% for the SVM.

For P41_01, CNN recall is 10.42% +/- 6.14 percentage points across seeds. For P41_02, CNN recall is 100.00% +/- 0.00 percentage points across seeds.

| Source | Size mm | DSP recall % | SVM recall % | CNN recall % | CNN BA % | CNN AUC |
|---|---|---|---|---|---|---|
| P41_01 | 2 | 0.00 | 98.72 | 10.42 +/- 6.14 | 55.21 +/- 3.07 | 1.0000 |
| P41_02 | 3 | 0.00 | 100.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.0000 |
| P41_03 | 26 | 100.00 | 100.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.0000 |
| P41_04 | 6 | 100.00 | 100.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.0000 |
| P41_05 | 17 | 100.00 | 100.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.0000 |
| P41_06_notch | 6 notch | 100.00 | 100.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.0000 |

All seed/fold outcomes: cnn_summary.csv. Interpretation limits: one repeated negative background; augmented positives; no independent specimen-level confidence intervals.
