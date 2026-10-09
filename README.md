# Noise-Augmented Urban Sound Classification under Matched and Mismatched Conditions

**ELEC5305 Speech and Audio Processing Project**<br>
**Student:** Tongtong Chen<br>
**Student ID:** 540334953

[Project proposal PDF](./ELEC5305_Project_Proposal_540334953.pdf)

## Research question

> To what extent does waveform-level noise augmentation improve an MFCC-SVM urban sound classifier when test noise is matched to, or different from, the noise used during training?

The project deliberately holds the feature/classification system fixed and treats the added-noise condition as the main experimental variable. The aim is not to obtain the highest possible UrbanSound8K accuracy, but to determine whether robustness learned from one added noise transfers to a different or previously unseen acoustic environment.

## Project progression

1. Reproduce the published UrbanSound8K MFCC-RBF-SVM baseline as closely as practical.
2. Analyse how added white, pink and environmental noise changes waveforms, spectra, log-Mel spectra and MFCCs.
3. Train models using original recordings and controlled noise augmentation.
4. Compare matched, mismatched and held-out test conditions at 20, 10 and 0 dB SNR.
5. Relate MFCC representation change to classification performance and class-level errors.

The term **original/unaugmented** is used instead of **clean** because UrbanSound8K contains real field recordings that may already include interfering sources.

## Published baseline reproduction

The baseline follows Salamon, Jacoby and Bello (2014):

- 44.1 kHz mono analysis rate
- 1024-sample window (approximately 23.2 ms) and 50% overlap
- 40 Mel bands from 0 to 22.05 kHz
- first 25 MFCC coefficients
- no pre-emphasis and no liftering
- RBF support vector machine
- official predefined ten-fold cross-validation
- preprocessing and feature selection fitted only on each training fold

The paper reports a 225-dimensional clip vector, but its prose can also be read as describing 275 values. The primary `paper225` implementation therefore uses seven statistics of each MFCC plus the mean first and second frame derivatives, giving `25 x 9 = 225`. A `literal275` option additionally includes the derivative variances. This ambiguity and the Python-versus-Essentia/Weka implementation differences are reported rather than hidden.

## Experimental conditions

Each noise-augmented strategy retains the original training clips and adds one deterministic augmented copy per clip. This keeps the augmented training-set size comparable across strategies.

| Training strategy | Added training material |
|---|---|
| `original` | Original UrbanSound8K audio only |
| `white` | White noise at a seeded random SNR from 0, 10 or 20 dB |
| `traffic` | DEMAND `STRAFFIC` at a seeded random SNR |
| `mixed` | A seeded choice of white, pink or `STRAFFIC` noise and SNR |

Every model is evaluated on original audio and the following added-noise conditions:

| Test condition | Role |
|---|---|
| White | Matched for the white model; seen by the mixed model |
| Pink | Mismatched for single-noise models; seen by the mixed model |
| DEMAND `STRAFFIC` | Matched for the traffic model; seen by the mixed model |
| DEMAND `PCAFETER` | Held-out environment, excluded from all training |

Noisy tests are generated at 20, 10 and 0 dB SNR. The first 60% of each environmental-noise recording supplies training excerpts, the final 30% supplies test excerpts, and a 10% temporal guard separates them. The held-out `PCAFETER` environment is never used for augmentation training.

## SNR mixing protocol

For original signal `s` and zero-mean noise `n`, full-clip RMS power is used:

```text
Ps = mean(s^2)
Pn = mean(n^2)
alpha = sqrt(Ps / (Pn * 10^(SNR_dB/10)))
y = s + alpha*n
```

Mixing is performed in floating point. If the mixture exceeds the output range, one common gain is applied to the entire mixture; no nonlinear clipping or limiting is used. Every random selection is derived from SID-based seed `540334953`, the clip identifier and the experimental condition.

## Repository structure

```text
configs/experiment.yaml       experiment and feature parameters
src/elec5305/features.py      log-Mel, MFCC and 225/275-D summaries
src/elec5305/noise.py         SNR mixing and noise partitioning
src/elec5305/baseline.py      published-baseline reproduction
src/elec5305/experiment.py    matched/mismatched cross-validation
src/elec5305/analysis.py      waveform/spectral/MFCC analysis
src/elec5305/plot_results.py  result plots and confusion matrices
scripts/smoke_test.py         complete synthetic end-to-end validation
tests/                        unit tests for features and SNR mixing
report/                       report source and experiment record
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Data layout

Download and extract UrbanSound8K without changing its official folds:

```text
data/UrbanSound8K/
  audio/fold1/ ... fold10/
  metadata/UrbanSound8K.csv
```

For DEMAND, select one channel from the official 48 kHz recordings and retain the source names:

```text
data/DEMAND/STRAFFIC.wav
data/DEMAND/PCAFETER.wav
```

The datasets are not redistributed in this repository. UrbanSound8K is subject to its original non-commercial licence, and DEMAND is distributed under CC BY-SA 3.0.

## Reproduce the baseline

```bash
elec5305-baseline \
  --dataset-root data/UrbanSound8K \
  --config configs/experiment.yaml \
  --output-dir results/baseline \
  --cache-dir data/cache \
  --n-jobs 4
```

This produces fold-level metrics, out-of-fold predictions, a normalised confusion matrix and a JSON manifest documenting the exact implementation and known reproduction differences.

## Analyse representation changes

Choose a representative UrbanSound8K clip and run:

```bash
elec5305-analyse \
  --audio data/UrbanSound8K/audio/fold1/EXAMPLE.wav \
  --class-label "siren" \
  --traffic-noise data/DEMAND/STRAFFIC.wav \
  --crowd-noise data/DEMAND/PCAFETER.wav \
  --output-dir results/signal_analysis
```

The command saves audio examples, noise power spectra, waveform/log-Mel/MFCC comparison figures and quantitative relative representation distances.

## Run the full experiment

```bash
elec5305-experiment \
  --dataset-root data/UrbanSound8K \
  --traffic-noise data/DEMAND/STRAFFIC.wav \
  --crowd-noise data/DEMAND/PCAFETER.wav \
  --config configs/experiment.yaml \
  --output-dir results/full_experiment \
  --cache-dir data/cache \
  --n-jobs 4

elec5305-plot-results --results-dir results/full_experiment
```

Primary outputs include accuracy and macro F1 across all official folds, per-class predictions, accuracy-versus-SNR curves, a noise-transfer heatmap and selected confusion matrices.

## Validation without downloading the datasets

The smoke test generates a small synthetic dataset and exercises the complete pipeline:

```bash
pytest -q
python scripts/smoke_test.py
```

Synthetic scores only verify software operation and must not be reported as research results.

## Reproducibility safeguards

- Official UrbanSound8K folds are never shuffled or replaced with random splits.
- Related slices therefore remain in the same fold.
- Scaling and feature selection are fitted only on the training portion of each fold.
- Augmentation and noise-excerpt selection are deterministic from recorded seeds.
- Training and test excerpts use non-overlapping regions of environmental recordings.
- `PCAFETER` is a completely unseen environment during training.
- Cached features, model files, predictions and run manifests preserve the experiment configuration.

## References

1. J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” *ACM Multimedia*, 2014. [DOI](https://doi.org/10.1145/2647868.2655045)
2. S. B. Davis and P. Mermelstein, “Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences,” *IEEE TASSP*, 1980. [DOI](https://doi.org/10.1109/TASSP.1980.1163420)
3. J. Dennis, H. D. Tran, and H. Li, “Spectrogram Image Feature for Sound Event Classification in Mismatched Conditions,” *IEEE Signal Processing Letters*, 2011. [DOI](https://doi.org/10.1109/LSP.2010.2100380)
4. J. Salamon and J. P. Bello, “Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification,” *IEEE Signal Processing Letters*, 2017. [DOI](https://doi.org/10.1109/LSP.2017.2657381)
5. J. Thiemann, N. Ito, and E. Vincent, “The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND),” *Proceedings of Meetings on Acoustics*, 2013. [DOI](https://doi.org/10.1121/1.4806631)
6. B. McFee et al., “A Software Framework for Musical Data Augmentation,” *ISMIR*, 2015. [Proceedings](https://archives.ismir.net/ismir2015/paper/000104.pdf)
7. K. J. Piczak, “Environmental Sound Classification with Convolutional Neural Networks,” *IEEE MLSP*, 2015. [DOI](https://doi.org/10.1109/MLSP.2015.7324337)
8. S. Chu, S. Narayanan, and C.-C. J. Kuo, “Environmental Sound Recognition with Time-Frequency Audio Features,” *IEEE TASLP*, 2009. [DOI](https://doi.org/10.1109/TASL.2009.2017438)

## Status

The codebase and synthetic validation are complete. Research results remain provisional until all ten official UrbanSound8K folds and the specified DEMAND recordings have been processed. The final report must use those real-data outputs, not smoke-test values.
