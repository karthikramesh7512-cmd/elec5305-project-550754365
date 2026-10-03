# DSP Baseline V1: post-hoc diagnostics

## Scope and preservation

All 7,000 experimental images (013–019) were examined. V1 code, thresholds,
predictions, summaries, figures and split CSVs remain unchanged. No selection
function was called. Existing validation candidates were read, and balanced
accuracy was checked at the already-saved thresholds. Metadata location/depth
were joined only AFTER pixel-only peak extraction. No SVM/CNN was implemented.

## Observations: continuous W=1 score distributions

Each image is counted once, not once per fold. Population standard deviation
uses ddof=0; quantiles use pandas linear interpolation. All V1 schemes selected W=1.
The approximately 9e-13 no-flaw standard deviation is floating-point roundoff;
its input scores are identical, so the mathematical standard deviation is zero.

| source | n | minimum | p25 | median | p75 | maximum | mean | std_population |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_flaw | 3558 | 3151.372023 | 3151.372023 | 3151.372023 | 3151.372023 | 3151.372023 | 3151.372023 | 0.000000 |
| P41_01 | 547 | 4583.179837 | 4611.722129 | 4630.784418 | 4648.529675 | 4909.091702 | 4633.800663 | 31.956441 |
| P41_02 | 543 | 6325.291403 | 6338.533825 | 6343.453138 | 6348.111394 | 6364.878168 | 6343.184271 | 7.636287 |
| P41_04 | 558 | 13938.366294 | 13959.466435 | 13964.386867 | 13968.756820 | 13981.376161 | 13963.944187 | 7.330739 |
| P41_06_notch | 599 | 7392.840322 | 7407.554607 | 7414.899613 | 7424.420872 | 7460.016833 | 7417.805296 | 14.210300 |
| P41_05 | 574 | 10232.792205 | 10246.811609 | 10251.313261 | 10256.753888 | 10274.197985 | 10251.543292 | 7.285621 |
| P41_03 | 621 | 14978.067378 | 14994.589565 | 15000.702804 | 15006.316051 | 15029.970962 | 15000.932688 | 9.274677 |

See `diagnostics/score_distributions_with_thresholds.png` for boxplots in size
order and overlays of the six unchanged validation-selected thresholds. These
are descriptive all-data diagnostics, not a new evaluation or tuning exercise.

## Observations: no-flaw maxima and raw data

There are 3558 no-flaw images. Unique raw full-image SHA-256 hashes:
1. Images matching reference `batch_013:4`:
3558/3558. Hashes cover the original
6,881,280 decompressed bytes (all 480 x 7168 int16 values), not the cropped ROI.
All no-flaw images, rather than just a sample, were hashed; per-image hashes
are saved in `diagnostics/peak_locations_and_hashes.csv`.

Reference SHA-256: `2fc5a1dd58b581fa3e383427e0bde5cc70885b249101be77f8994acb25c7ae4a`.

Unique W=1 scores after rounding to 3/6/9/12 decimal places: {'3': 1, '6': 1, '9': 1, '12': 1}.
Peak coordinates are zero-based; time is the original image index (ROI offset added).
Argmax uses the first occurrence in C order; exact tie counts are also saved.

| peak_scan | peak_time | n_samples |
| --- | --- | --- |
| 383 | 2176 | 3558 |

Representative raw-RF/envelope neighbourhoods and traces are in
`diagnostics/no_flaw_peak_batch_013.png`, `no_flaw_peak_batch_016.png`, and
`no_flaw_peak_batch_019.png`. They are selected by batch/row, not peak strength.

## Observations: positive peak association

The diagnostic compares positive images with reference `batch_013:4` in
the fixed ROI. The changed-pixel bounding box is measured from pixel differences,
not imposed from metadata. Reported offsets are peak minus metadata coordinates.

| source | n | min_scan_offset | max_scan_offset | min_time_offset | max_time_offset | fraction_peak_in_changed_bbox | fraction_peak_pixel_changed | fraction_origin_matches_metadata |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P41_01 | 547 | 9.268582 | 20.998855 | 298.538531 | 437.538531 | 0.983547 | 0.983547 | 0.996344 |
| P41_02 | 543 | 12.000553 | 12.998031 | 163.833890 | 167.833890 | 1.000000 | 1.000000 | 0.998158 |
| P41_03 | 621 | 32.000579 | 32.997873 | 218.031981 | 218.031981 | 1.000000 | 1.000000 | 0.998390 |
| P41_04 | 558 | 12.002159 | 12.998147 | 193.876445 | 197.876445 | 1.000000 | 1.000000 | 0.998208 |
| P41_05 | 574 | 25.001734 | 25.999856 | 230.512657 | 230.512657 | 1.000000 | 1.000000 | 1.000000 |
| P41_06_notch | 599 | 16.000780 | 16.999702 | 177.962602 | 184.962602 | 1.000000 | 1.000000 | 1.000000 |

Of 3,442 positive peaks, 3,433 are inside the changed-pixel bounding box.
The nine exceptions are all P41_01, exactly one time sample beyond its changed
region (peak time 2119 vs changed-region end 2118), and within its scan span.
They remain immediately adjacent to the inserted region. Hilbert envelopes depend
on the whole A-scan, so a peak outside exact changed-pixel support is possible.
This observation alone does not diagnose an augmentation artefact. See
`diagnostics/positive_peak_bbox_exceptions.csv`.

The difference origin matches integer-truncated metadata for 3,437/3,442 positives.
For five mismatches, the first changed scan is 2?5 positions after the metadata
origin, while the time origin agrees. Leading unchanged pixels or background
 overlap may explain this; generation details remain unresolved.

All computed candidate scores matched frozen V1 scores within rtol=1e-12,
atol=1e-9. Metadata did not participate in any of those computations.

## Observations: the four window sizes

Saved validation balanced accuracy (not newly optimized):

| scheme | fold | 1 | 3 | 5 | 7 |
| --- | --- | --- | --- | --- | --- |
| ordinary | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| source_holdout | 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| source_holdout | 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| source_holdout | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| source_holdout | 4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| source_holdout | 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| source_holdout | 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

`diagnostics/window_validation_behavior.csv` reports each unchanged candidate
threshold, positive minimum, negative maximum, and their gap. Full per-source
score behavior for every W is in `diagnostics/window_score_behavior.csv`.
The smallest observed validation separation gap across the 28 saved candidates
is 952.335804. Positive gaps explain why all four W values
can perfectly separate validation classes. W=1 was selected by the previously
specified smallest-W tie rule, not evidence that it performs better.

Median score by source and window:

| source | 1 | 3 | 5 | 7 |
| --- | --- | --- | --- | --- |
| no_flaw | 3151.372023 | 3036.055609 | 2938.121381 | 2940.270536 |
| P41_01 | 4630.784418 | 4432.205740 | 4121.808621 | 3912.223277 |
| P41_02 | 6343.453138 | 6143.621969 | 5768.356805 | 5345.806007 |
| P41_04 | 13964.386867 | 13383.542196 | 12423.688009 | 11310.276636 |
| P41_06_notch | 7414.899613 | 7235.379996 | 7071.075469 | 6800.733449 |
| P41_05 | 10251.313261 | 10050.689107 | 9817.999624 | 9485.726716 |
| P41_03 | 15000.702804 | 14180.521253 | 13239.370199 | 11917.936820 |

The no-flaw maximum is slightly higher at W=7 than W=5: different complete
windows can maximize the average, so monotonic decrease is not guaranteed.
For no-flaw images, the maximizing spatial window centres are scan 383 for W=1/3
and scan 435 for W=5/7. A smoothed score averages line peaks and therefore has no
single corresponding time coordinate when W is greater than one.

## Supported interpretations

The single shared raw hash across all 3,558 no-flaw images explains identical scores directly: repeated
pixel data necessarily give the same deterministic envelope and maximum. A common
background response at a fixed scan/time coordinate is consistent with a fixed
echo inherited from the acquisition/background. Repeated augmented images are
not independent acquisitions and cannot establish the physical origin of that echo.
No response is classified here as an artefact.

Where the peak lies in the changed region and the region's origin matches metadata,
the maximum is spatially associated with the inserted flaw. Metadata describes an
insertion origin in this dataset; a nonzero peak offset is not itself a detection
localization error. A rectangular bounding-box association is not a physical flaw mask.

Spatial averaging changes score magnitudes without resolving the validation tie.
The V1 threshold policy chose the highest equally optimal threshold. Perfect
validation separation leaves a range of equally good operating points, so this
tie policy can matter for lower-amplitude unseen sources. It is left unchanged.

## Unresolved uncertainties

The common background echo cannot be uniquely attributed to a system response,
interface/backwall, geometry or material structure from these files alone.
Independent acquisitions, calibration, geometry and instrument details are needed.
Hash equality establishes equality of decoded images, not why they were repeated.
Association with an inserted region does not establish physical depth accuracy.
No new operating point or W is proposed from these post-hoc diagnostics.
