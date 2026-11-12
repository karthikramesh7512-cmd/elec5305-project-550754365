# Final-stage DSP and DSP plus CNN comparison

## Scope

This extension adds a compact CNN classifier downstream of the shared DSP front end. It does not change or rerun the saved deterministic detector or engineered-feature SVM. The question is whether a learned convolutional decision rule improves detection of previously unseen flaw sources relative to the frozen validation-thresholded DSP detector. Improvement over the existing SVM is a separate comparison; improvement is not assumed.

## Fixed protocol

- Data: the same 7,000 experimental rows, batches 013-019; raw SHA-256 must match the original feature audit for every sample.
- Evaluation: the six saved source-holdout folds are primary. The saved ordinary 70/15/15 split is a reference only. No split is regenerated.
- DSP front end: columns 1100:3100; DC removal independently along each A-scan; Hilbert envelope along the time/depth axis, using the original double-precision computation.
- CNN representation: non-overlapping 8 by 16 maximum pooling of the 480 by 2000 envelope into a 60 by 125 map, followed by fixed log1p compression. This retains local peaks while reducing computational cost. The CNN therefore uses a compressed DSP representation, not the original full-resolution RF image.
- Scaling: one scalar mean and standard deviation fitted to the training maps separately for every evaluation. No validation or test map contributes to these fitted parameters.
- Architecture: three padded 3 by 3 convolution layers with 8, 16 and 32 channels and ReLU activations; 2 by 2 max pooling after the first two layers; concatenated global average and maximum pooling; one linear output logit. No pretrained weights or metadata input.
- Training: Adam, learning rate 0.001, weight decay 0.0001, unweighted binary cross entropy with logits, batch size 64, at most 20 epochs. No augmentation or hyperparameter search. Fixed seeds 42, 43 and 44 measure initialization/training variation.
- Checkpoint selection: highest validation balanced accuracy, then lowest validation binary cross entropy; earliest checkpoint on an exact tie. Stop after five consecutive epochs without improvement in that ordering. Classification threshold remains fixed at sigmoid probability 0.5.
- Freeze all 21 selected checkpoints before evaluating any test set. No train-plus-validation refit and no test-based tuning or architecture revisions.
- Report every seed and fold, mean and sample standard deviation across seeds, confusion counts, recall, specificity, precision, F1, balanced accuracy and ROC-AUC. Compare against the exact saved baseline rows for the same scheme and fold.
- Save training curves, fit/validation IDs, selected epochs, weights, preprocessing configuration and provenance. Exclude the local envelope-map cache from GitHub.

## Interpretation limits

All 3,558 no-flaw rows repeat one raw background. Negative rows are not independent acquisitions, and specificity cannot demonstrate performance on unseen no-flaw backgrounds. Related augmented positives also limit independence. Source holdout tests exclusion of P41 source labels, not new specimens or acquisition systems. Seed variability is not a confidence interval for independent physical flaws. Full-resolution processing and robustness to acquisition noise are not established by this experiment.

No CNN result may be described as an improvement until the fixed evaluation has completed. Report ties or deterioration explicitly. Original model outputs and all 105 protected data/results files remain byte-for-byte preserved.
