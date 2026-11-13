
# Results Directory

This directory contains verified results from the completed ELEC5305 project.

## Full Experiment

The `full_experiment/` directory contains outputs from the official 10-fold UrbanSound8K evaluation.

The experiment evaluated four training strategies across 13 test conditions per fold, producing 520 fold-level metric records.

### Available Files

- `aggregate_metrics.csv`: Aggregated classification performance.
- `all_fold_metrics.csv`: Fold-level evaluation results.
- `experiment_manifest.json`: Experiment configuration and metadata.
- `accuracy_vs_snr.png`: Accuracy across noise types and SNR levels.
- `transfer_heatmap_0db.png`: Noise transfer performance at 0 dB SNR.
- `confusion_mixed_crowd_0_dB.png`: Mixed-model confusion matrix under held-out crowd noise.
- `confusion_original_original_original.png`: Original-model confusion matrix.

## Main Finding

Mixed noise augmentation improved average noisy-condition accuracy from 50.23% to 59.91%, a gain of 9.68 percentage points.

However, Traffic augmentation performed better on the held-out crowd environment.

## Reproducibility

Source code, experiment configuration, and execution instructions are available in the repository root.

Large prediction archives, model files, feature caches, and the original audio datasets are excluded from version control.

## Video Demonstration

https://youtu.be/T1r7gkJSBcs
