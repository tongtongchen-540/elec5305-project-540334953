# Noise-Augmented Urban Sound Classification under Matched and Mismatched Conditions

**ELEC5305 Speech and Audio Processing Project**

**Student:** Tongtong Chen
**Student ID:** 540334953
**GitHub username:** tongtongchen-540

[Download the Project Proposal PDF](./ELEC5305_Project_Proposal_540334953.pdf)

## Project Overview

Urban sound classifiers recognise events such as sirens, dog barks, car horns, drills, and engines. However, recordings made in real environments may contain traffic, crowds, wind, or sensor noise that was absent during training. This difference between training and deployment conditions is known as a noise mismatch.

This project will reproduce an MFCC-SVM classifier using the UrbanSound8K dataset and then introduce a controlled modification: adding selected noises to the training audio. Clean, single-noise, and mixed-noise training strategies will be evaluated using both familiar and previously unseen noise conditions.

## Research Question

> To what extent does waveform-level noise augmentation improve an MFCC-SVM urban sound classifier when test noise is matched to, or different from, the noise used during training?

The project investigates whether noise augmentation produces general robustness or simply makes a classifier more specialised for the particular noise used during training.

## Dataset

The project will use the **UrbanSound8K** dataset, which contains 8,732 labelled audio excerpts across ten urban sound classes:

1. Air conditioner
2. Car horn
3. Children playing
4. Dog bark
5. Drilling
6. Engine idling
7. Gun shot
8. Jackhammer
9. Siren
10. Street music

Real environmental noise recordings will be obtained from the **DEMAND** acoustic noise database.

## Proposed Method

The main signal-processing and classification pipeline is:

1. Convert audio to mono and resample it to 22.05 kHz.
2. Divide each signal into overlapping short-time frames.
3. Compute the Short-Time Fourier Transform (STFT).
4. Generate a 40-band log-Mel spectrogram.
5. Extract MFCC, delta, and delta-delta features.
6. Summarise the features using their mean and standard deviation.
7. Scale the features using statistics fitted only on training data.
8. Train a radial-basis-function Support Vector Machine (SVM).
9. Evaluate the classifier on clean and noisy recordings.

The official ten UrbanSound8K folds will be preserved to prevent related recordings from appearing in both training and testing data.

## Experimental Design

Four training strategies will be compared:

| Model               | Training condition                          |
| ------------------- | ------------------------------------------- |
| Baseline            | Clean audio only                            |
| White-noise model   | White-noise augmentation                    |
| Traffic-noise model | DEMAND traffic-noise augmentation           |
| Mixed-noise model   | White, pink, and traffic-noise augmentation |

Each system will be tested using:

* Clean audio
* White noise
* Pink noise
* Traffic noise
* Held-out crowd noise

Noisy recordings will be generated at **20 dB, 10 dB, and 0 dB SNR**.

White-to-white and traffic-to-traffic tests represent matched conditions. Other combinations represent mismatched conditions. Crowd noise will be completely excluded from training and used to measure generalisation to an unseen environment.

## Evaluation

Performance will be evaluated using:

* Classification accuracy
* Macro F1-score
* Per-class recall
* Confusion matrices
* Accuracy-versus-SNR curves
* Training-noise versus test-noise heatmaps
* Runtime and feature dimensionality

The analysis will examine the improvement over clean-only training, the difference between matched and mismatched conditions, and any reduction in clean-audio performance.

## Expected Outcomes

A single-noise model may perform best when its training and testing noise conditions match. Mixed-noise training is expected to be less specialised but potentially more stable when encountering unseen noise.

A result showing little or no improvement would also be meaningful because it would reveal the limitations of simple waveform-level augmentation. The final discussion will relate the results to MFCC noise sensitivity, spectral masking, sound-class similarity, and previous environmental sound classification research.

## Project Deliverables

* Working and reproducible Python code
* Configuration files and saved experimental results
* Performance graphs, tables, and confusion matrices
* Written final project report
* Audio demonstrations
* Brief project demonstration video

## References

1. J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” *Proceedings of the 22nd ACM International Conference on Multimedia*, 2014. [DOI](https://doi.org/10.1145/2647868.2655045)

2. S. B. Davis and P. Mermelstein, “Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1980. [DOI](https://doi.org/10.1109/TASSP.1980.1163420)

3. J. Dennis, H. D. Tran, and H. Li, “Spectrogram Image Feature for Sound Event Classification in Mismatched Conditions,” *IEEE Signal Processing Letters*, 2011. [DOI](https://doi.org/10.1109/LSP.2010.2100380)

4. J. Salamon and J. P. Bello, “Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification,” *IEEE Signal Processing Letters*, 2017. [DOI](https://doi.org/10.1109/LSP.2017.2657381)

5. J. Thiemann, N. Ito, and E. Vincent, “The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND),” *Proceedings of Meetings on Acoustics*, 2013. [DOI](https://doi.org/10.1121/1.4806631)
