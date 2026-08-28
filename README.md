# Ultrasonic Crack Detection and Localisation in Steel

ELEC5305 Project – University of Sydney

## Overview

This project investigates the use of Digital Signal Processing (DSP) and Machine Learning (ML) for detecting and localising cracks and other defects in steel using ultrasonic inspection data.

The project will first develop a conventional DSP-based detection system and then compare its performance with machine-learning approaches.

The main objective is to determine whether machine learning can improve ultrasonic defect detection, particularly when the signal quality is reduced by noise.

## Project Approach

The project will compare three main approaches:

1. Conventional DSP-based defect detection
2. DSP feature extraction combined with a Support Vector Machine (SVM)
3. A Convolutional Neural Network (CNN) trained directly on ultrasonic B-scan data

The general processing pipeline is:

Ultrasonic Data  
→ Signal Processing  
→ Feature Extraction  
→ Defect Detection / Localisation

For Full Matrix Capture (FMC) data:

FMC Data  
→ Signal Processing  
→ Total Focusing Method (TFM)  
→ Ultrasonic Image Reconstruction  
→ Defect Localisation

## Digital Signal Processing

The initial DSP system will investigate techniques including:

- Signal normalisation
- Filtering and denoising
- Time and spatial gating
- Envelope and amplitude processing
- Peak detection
- RMS amplitude
- Signal energy
- Time-of-flight analysis
- Threshold-based defect detection
- FMC processing
- Total Focusing Method beamforming

This system will provide an interpretable baseline against which the machine-learning methods can be compared.

## Machine Learning

### Support Vector Machine

A Support Vector Machine will be trained using features extracted from the DSP pipeline.

Potential features include:

- Peak amplitude
- RMS amplitude
- Signal energy
- Peak position
- Spatial statistics
- Defect-response characteristics

### Convolutional Neural Network

A compact CNN will also be trained directly using ultrasonic B-scan data.

The final comparison will therefore be:

DSP vs DSP + SVM vs CNN

## Datasets

### NDT_ML_Flaw

Primary dataset for machine-learning experiments.

The dataset contains approximately 17,000 ultrasonic samples across measured and simulated flaw batches.

Metadata includes:

- Flaw presence
- Flaw depth
- Flaw location
- Flaw size
- Flaw type

Dataset:

https://github.com/koomas/NDT_ML_Flaw

### ML-NDT

Secondary dataset containing approximately 20,000 augmented phased-array ultrasonic samples generated from measured thermal-fatigue crack signals.

Dataset:

https://github.com/iikka-v/ML-NDT

### University of Strathclyde FMC Data

Experimental Full Matrix Capture datasets containing known flaws and reflectors will be used for DSP, TFM reconstruction and defect-localisation experiments.

## Evaluation

The defect-detection methods will be evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

Defect localisation will be evaluated using the difference between the estimated and known defect position.

The robustness of each method will also be tested under different signal-to-noise ratio conditions.

Example SNR levels:

- 20 dB
- 15 dB
- 10 dB
- 5 dB
- 0 dB

## Data Splitting

Care will be taken to prevent data leakage.

Where possible, samples originating from the same physical flaw or augmentation batch will remain within the same training, validation or test partition.

This prevents closely related versions of the same defect from appearing in both the training and testing datasets.

## Tools

The project will primarily use Python.

Planned libraries include:

- NumPy
- SciPy
- Matplotlib
- Pandas
- scikit-learn
- PyTorch

## Expected Outcomes

The final project is expected to include:

- DSP-based ultrasonic defect detection
- FMC/TFM defect localisation
- Ultrasonic A-scan and B-scan visualisation
- DSP feature extraction
- SVM-based classification
- CNN-based classification
- Comparison of DSP, SVM and CNN performance
- Noise robustness analysis
- Performance metrics and visualisations

If time permits, equivalent flaw-size estimation will also be investigated.

## Project Status

Current stage: Project proposal and initial dataset investigation.

The repository will be updated throughout the semester as the DSP pipeline, machine-learning models and evaluation results are developed.

## References

1. I. Virkkunen, T. Koskinen, O. Jessen-Juhler, and J. Rinta-aho, "Augmented Ultrasonic Data for Machine Learning," Journal of Nondestructive Evaluation, vol. 40, art. 4, 2021.

2. T. Koskinen, I. Virkkunen, O. Siljama, and O. Jessen-Juhler, "The Effect of Different Flaw Data to Machine Learning Powered Ultrasonic Inspection," Journal of Nondestructive Evaluation, vol. 40, art. 24, 2021.

3. C. Holmes, B. W. Drinkwater, and P. D. Wilcox, "Post-processing of the full matrix of ultrasonic transmit-receive array data for non-destructive evaluation," NDT & E International, vol. 38, no. 8, pp. 701–711, 2005.

4. C. Holmes, B. W. Drinkwater, and P. D. Wilcox, "The post-processing of ultrasonic array data using the total focusing method," Insight – Non-Destructive Testing and Condition Monitoring, vol. 46, no. 11, pp. 677–680, 2004.

## Author

Karthik Ramesh  
ELEC5305 – University of Sydney

GitHub Repository:

https://github.com/karthikramesh7512-cmd/elec5305-project-550754365
