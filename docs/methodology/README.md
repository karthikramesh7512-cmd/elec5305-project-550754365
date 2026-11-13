# Methods and reproducibility

The following records document the datasets, evaluation design and baseline methods:

- [Source and image-identity audit](../../data/source_flaw_audit_notes.md)
- [Frozen evaluation splits](../../data/splits/README.md)
- [Data inspection](../../results/data_inspection/inspection_report.md)
- [DSP method](../../results/dsp_baseline/README.md)
- [DSP results](../../results/dsp_baseline/RESULTS.md)
- [DSP diagnostics](../../results/dsp_baseline/dsp_diagnostics.md)
- [SVM methods, features and results](../../results/svm/svm_report.md)

Run repository commands from the project root using the paths in the main README.
All 3,558 experimental no-flaw rows repeat one raw background. Specificity therefore does not establish performance on independent normal acquisitions.

## Reproducibility records

[Integrity records](packaging_preservation.json) contain source and result hashes. The saved models, numerical tables and fixed split manifests support independent verification.

## CNN and supplementary studies

- [Fixed CNN protocol](final_cnn_protocol.md)
- [CNN evaluation outputs](../../results/dsp_cnn/)
- [Noise, simulation-transfer and measured-FMC methods](completed_extensions.md)
- [Research report](../ELEC5305_Final_Report.pdf)
- [Project demonstration](../video/project_demo.mp4)

[Pre-extension result hashes](pre_extension_preservation.json) support the saved-result verification script. The supplementary outputs occupy separate result directories.
