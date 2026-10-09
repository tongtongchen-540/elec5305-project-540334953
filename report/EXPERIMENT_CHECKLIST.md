# Official experiment checklist

## Before running

- [ ] UrbanSound8K is extracted under `data/UrbanSound8K`.
- [ ] `metadata/UrbanSound8K.csv` contains 8,732 rows.
- [ ] Audio folders `fold1` to `fold10` are unchanged.
- [ ] A single DEMAND `STRAFFIC` channel is saved as `data/DEMAND/STRAFFIC.wav`.
- [ ] A single DEMAND `PCAFETER` channel is saved as `data/DEMAND/PCAFETER.wav`.
- [ ] Dataset licences and citations have been recorded.
- [ ] `pytest -q` and `python scripts/smoke_test.py` pass.

## Baseline gate

- [ ] Run all ten official folds.
- [ ] Inspect `fold_metrics.csv` for one unusually easy or difficult fold.
- [ ] Compare mean accuracy with the published MFCC-SVM result.
- [ ] Record all differences from the Essentia/Weka implementation.
- [ ] Do not continue to the full matrix until the baseline is plausible.

## Noise protocol gate

- [ ] Achieved SNR differs from the target by less than 0.01 dB in spot checks.
- [ ] Training and test environmental-noise offsets lie in disjoint time regions.
- [ ] `PCAFETER` is absent from all training caches and manifests.
- [ ] Saved audio examples contain no clipping distortion.
- [ ] White and pink power spectra have the expected qualitative shapes.

## Full experiment

- [ ] Four training strategies complete for all ten folds.
- [ ] Original plus 12 noisy test conditions complete for every strategy.
- [ ] Every row has accuracy, macro F1 and macro recall.
- [ ] Out-of-fold predictions cover every clip exactly once per model/condition.
- [ ] Accuracy-SNR curves, transfer heatmap and selected confusion matrices render correctly.
- [ ] Representation-distance table is linked to the performance analysis.

## Reporting

- [ ] No synthetic smoke-test value appears in the report.
- [ ] Tables state mean and standard deviation across ten folds.
- [ ] Figures include axes, units, legends and readable class labels.
- [ ] Discussion addresses original performance, matched gain, mismatched gain and held-out gain.
- [ ] GitHub instructions run successfully in a fresh environment.
- [ ] Report and GitHub site link to the final video.
