# Noise-Augmented Urban Sound Classification under Matched and Mismatched Conditions

> Working report source. Replace every `RESULT REQUIRED` marker only with values produced by the official ten-fold experiment. Synthetic smoke-test scores must never be reported as research results.

## Abstract

State the research question, published-baseline reproduction, four training strategies, four added-noise tests at three SNRs, principal ten-fold result, and the conclusion about matched-to-mismatched transfer in approximately 150-200 words.

## 1. Introduction

- Motivate reliable recognition of urban sound events under additional acoustic interference.
- Explain that UrbanSound8K recordings are real field recordings, so use **original/unaugmented**, not **clean**.
- State the single research question verbatim.
- Clarify that the objective is controlled understanding, not maximum benchmark accuracy.
- Summarise the reproduction component, modification, signal-processing analysis and main contribution.

## 2. Literature Review

### 2.1 Conventional environmental-sound classification

Discuss the UrbanSound8K taxonomy, official folds and the 2014 MFCC-SVM baseline. Explain the leakage risk from slices originating from the same Freesound recording.

### 2.2 MFCCs and additive interference

Explain framing, STFT, Mel filtering, logarithmic compression and DCT. Discuss why additive interference alters Mel-band energy and cepstral coefficients, especially for broadband and continuous sound classes.

### 2.3 Matched and mismatched acoustic conditions

Review evidence that classifiers degrade when evaluation conditions differ from development conditions. Define matched, mismatched, seen-in-mixture and held-out conditions used in this project.

### 2.4 Environmental-sound data augmentation

Review Salamon and Bello, MUDA and other relevant augmentation studies. Compare advantages and disadvantages of waveform augmentation, feature-domain augmentation, and neural representation learning. Explain why this project retains an MFCC-SVM.

### 2.5 Research gap

State that prior work shows augmentation can help, but the present small study focuses on whether robustness learned from one added-noise type transfers to a distinct and previously unseen environment while the feature/classifier system is fixed.

## 3. Methodology

### 3.1 Datasets

Report UrbanSound8K version, licence, 8,732 clips, ten classes and official folds. Identify DEMAND environments exactly as `STRAFFIC` and `PCAFETER`, the selected channel, original sample rate and duration.

### 3.2 Published baseline reproduction

Report:

- 44.1 kHz mono analysis rate;
- 1024-sample Hann frame and 512-sample hop;
- 40 Mel bands from 0 to 22.05 kHz;
- first 25 MFCCs, no pre-emphasis and no liftering;
- nine clip statistics per coefficient for the 225-dimensional interpretation;
- training-fold standardisation and correlation-aware selection;
- RBF SVM parameters;
- official ten-fold evaluation.

Explicitly discuss the paper's 225/275-dimensional wording ambiguity and the Essentia/Weka versus librosa/scikit-learn implementation differences.

### 3.3 Noise-mixing protocol

Define:

\[
P_s=\frac{1}{N}\sum_{i=1}^N s_i^2, \qquad
P_n=\frac{1}{N}\sum_{i=1}^N n_i^2
\]

\[
\alpha=\sqrt{\frac{P_s}{P_n10^{\mathrm{SNR}_{dB}/10}}}, \qquad
y=s+\alpha n.
\]

Explain DC removal for power measurement, full-clip RMS, equal-length noise excerpts, common-gain clipping prevention, fixed seeds, the 60% training/10% guard/30% test partition, and complete exclusion of `PCAFETER` from training.

### 3.4 Training and test conditions

Describe original, white, traffic and mixed training. Explain that every augmented strategy adds one augmented copy per original training clip, ensuring equal training-set size. Describe original plus white, pink, `STRAFFIC` and `PCAFETER` testing at 20, 10 and 0 dB.

### 3.5 Signal-processing analysis

Define waveform, Welch power spectrum, log-Mel and MFCC visualisations. Define relative representation distance:

\[
D(M,\widetilde{M})=\frac{\lVert\widetilde{M}-M\rVert_F}{\lVert M\rVert_F}.
\]

Explain how this is related to accuracy loss across noise type and SNR.

### 3.6 Evaluation

Report accuracy, macro F1, per-class recall, ten-fold mean and standard deviation, confusion matrices, accuracy-SNR curves, transfer heatmaps, runtime and selected feature count. All comparisons must use out-of-fold predictions.

## 4. Results

### 4.1 Reproduction result

**RESULT REQUIRED:** ten-fold baseline accuracy and macro F1, fold variability, selected feature count and comparison with the published result. Do not describe an implementation difference as an improvement.

### 4.2 Signal and representation changes

Insert `noise_power_spectra.png`, representative `representation_*.png` figures and the distance table. Explain the frequency-shape differences between white, pink, `STRAFFIC` and `PCAFETER` noise and how MFCC distance changes with SNR.

### 4.3 Overall noise robustness

Insert `accuracy_vs_snr.png` and the aggregate accuracy/macro-F1 table. Identify the strongest and weakest strategies at each test noise and SNR.

### 4.4 Matched versus mismatched transfer

Insert `transfer_heatmap_0db.png`. Compute and report:

- matched augmentation gain over original-only training;
- mismatched augmentation gain;
- held-out `PCAFETER` gain;
- any loss on original recordings.

### 4.5 Class-level errors

Use per-class recall and confusion matrices to explain which classes benefit or degrade. Compare findings with the original paper's confusions among air conditioner/engine idling, jackhammer/drilling and children playing/street music.

## 5. Discussion

- Answer the research question directly.
- Distinguish noise-specific robustness from transferable robustness.
- Relate classification performance to log-Mel/MFCC representation distance.
- Explain unexpected or null results rather than hiding them.
- Compare findings with previous augmentation and mismatch literature.
- Discuss limitations: implementation differences from Essentia/Weka, single classifier family, noise-source coverage, full-clip SNR, mono conversion and computational budget.
- Suggest focused future work, such as more noise environments, active-level SNR, class-conditional augmentation, temporal features or one neural comparison.

## 6. Conclusion

In one concise section, state what was reproduced, what was changed, the strongest evidence obtained, how fully the research question was answered and the main limitation.

## Reproducibility and Data Availability

Link the GitHub Pages site, repository, report PDF and demonstration video. State the Python version, package environment, seed, configuration path, dataset licences and exact commands used. Do not redistribute UrbanSound8K or DEMAND audio in the repository.

## References

Use one consistent IEEE-style bibliography. Include at least the eight references listed in the revised proposal and add any further papers that are substantively analysed in the literature review.
