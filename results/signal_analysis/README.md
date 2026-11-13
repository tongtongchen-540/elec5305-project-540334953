# Supplementary Signal Analysis

This directory provides a compact, verifiable supplement for the report's Section 4.5.

## Files

- [all_representation_distances.csv](all_representation_distances.csv): 48 single-clip perturbation measurements from the completed signal-analysis run, rounded to six decimal places for repository display.
- [selected_audio_manifest.csv](selected_audio_manifest.csv): The four selected Fold 1 example recordings.
- [mfcc_distance_vs_snr.svg](mfcc_distance_vs_snr.svg): Mean relative MFCC distance across the four selected recordings, by noise type and test SNR.

The 48 records correspond to four clips (siren, street music, engine idling, gun shot) × four added noises (white, pink, DEMAND STRAFFIC, DEMAND PCAFETER) × three nominal SNRs (20, 10, 0 dB). The measurements illustrate within-recording representation changes and **are not classification scores or 48 independent clips**.

The original waveform, log-Mel, MFCC and noise-spectrum PNGs can be regenerated with the `elec5305-analyse` command documented in the [project README](../../README.md). The source UrbanSound8K and DEMAND recordings are not redistributed here.

The full ten-fold classification results are separate and available in [../full_experiment/](../full_experiment/).
