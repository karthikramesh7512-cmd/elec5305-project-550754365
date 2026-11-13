# Completed supplementary experiments

The supplementary studies examine additive-noise sensitivity, simulation-trained transfer and measured-FMC imaging. The primary research question concerns the effect of a CNN decision stage on unseen-source flaw detection. The imaging pilot evaluates a separate endpoint.

## Saved-result checks (no raw data or training)

```powershell
python -m pip install -r requirements-cnn.txt
python src/demo_cnn.py
python src/verify_cnn_results.py
python src/verify_extensions.py
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v
```

## Noise stress test

Results: [tables and predictions](../../results/noise_robustness/) and [figure](../../results/noise_robustness/noise_performance.png).
Five added ROI RMS SNRs, three Gaussian draws, frozen DSP decisions and three CNN seeds. SVM is not included. Duplicate images share the same perturbation. Synthetic noise does not produce new physical normal acquisitions. Error bars describe perturbation/seed variation, not specimen confidence.

To reproduce in a fresh checkout containing the primary saved CNN outputs, first run primary preprocessing/training as documented in the README if the ignored `.cnn_cache` is absent, then:

```powershell
python src/noise_robustness.py --dataset-dir "PATH/TO/datasets"
```

Completed-result guards prevent accidental overwrites. Do not remove guards to retune on test results. The raw streams and ignored map caches are large and are not distributed.

## Simulation-trained transfer

Results: [source splits, selected models, predictions and audit](../../results/simulated_to_real/).
Train on simulated batches 201-207; validate on 208; test on 209-210 and all 7,000 experimental target rows. Each image is 480 by 7168 little-endian int16; 1,000 images per stream. All model selection occurs in simulation. Size-code-to-mm mapping remains unverified.

**The 4,982 simulated and 3,558 experimental negatives share one exact image.** The audit discloses this before fitting; no positive hash overlaps. Perfect experimental-target scores therefore do not establish independent-background transfer. Keep this caveat beside any metric or figure.

```powershell
python src/complete_extensions.py transfer --dataset-dir "PATH/TO/datasets"
```

Use a fresh copy without the completed simulated_to_real results. `--resume-preprocessing` is for an interrupted local preprocessing pass with the validated cache; it is not needed on a fresh run.

## Measured FMC/TFM

Results: [initial and calibrated reconstructions and provenance](../../results/fmc_tfm/).
Acquire the public [Strathclyde FMC dataset](https://doi.org/10.15129/a7da5071-0436-4913-8f0e-de78f5ebccd6) separately. Attribution: Timothy Lardner (creator), Katherine M. M. Tant (contributor), University of Strathclyde, 2015, CC BY 4.0. The dataset is a welded 316L plate with a nominal 6 mm lack-of-fusion flaw. This project derives images and calibration tables; it does not redistribute the original 260 MB MAT file.

```powershell
python src/complete_extensions.py tfm --fmc "PATH/TO/FMC_2012_06_12_at_13_09.mat"
python src/calibrated_tfm.py --fmc "PATH/TO/FMC_2012_06_12_at_13_09.mat"
```

Actual MATLAB axis order is time, transmitter, receiver. Full-aperture homogeneous longitudinal TFM uses 5,820 m/s. The 1.21014 microsecond time-origin calibration uses remote-element backwall arrivals, not the crack-tip position. Nominal-tip distances are 5.94 mm initially and 3.91 mm calibrated, but the calibrated ROI peak is on a boundary. This is a failed reliable-tip-localisation check, not a crack-sizing result. Reference geometry is approximate and the ROI uses metadata.

## External validation still unavailable

There are no independently acquired additional no-flaw specimens in the supplied project data. Neither synthetic perturbation nor the measured FMC flawed plate supplies a justified independent normal detection population. Such measurements require new data acquisition or a separately validated compatible dataset; they cannot be fabricated. The report treats this as a scientific limitation.
