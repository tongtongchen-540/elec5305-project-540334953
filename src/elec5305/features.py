from __future__ import annotations

from dataclasses import asdict

import librosa
import numpy as np
from scipy.stats import kurtosis, skew

from .config import FeatureConfig


def log_mel_spectrogram(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute the log-Mel power representation used before the DCT."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < config.frame_length:
        audio = np.pad(audio, (0, config.frame_length - audio.size))

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.frame_length,
        window="hann",
        center=False,
        power=2.0,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        htk=False,
        norm="slaney",
    )
    return librosa.power_to_db(mel_power, ref=1.0, top_db=None)


def mfcc_matrix(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute a 25 x frames MFCC matrix approximating Salamon et al. (2014).

    The paper used Essentia. This implementation uses librosa and records all
    parameters so the numerical implementation difference is explicit.
    """
    log_mel = log_mel_spectrogram(audio, config)
    matrix = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=config.n_mfcc,
        dct_type=2,
        norm="ortho",
        lifter=0,
    )
    return np.asarray(matrix, dtype=np.float64)


def _base_statistics(matrix: np.ndarray) -> list[np.ndarray]:
    return [
        np.min(matrix, axis=1),
        np.max(matrix, axis=1),
        np.median(matrix, axis=1),
        np.mean(matrix, axis=1),
        np.var(matrix, axis=1),
        skew(matrix, axis=1, bias=False),
        kurtosis(matrix, axis=1, fisher=True, bias=False),
    ]


def summarise_mfcc(matrix: np.ndarray, variant: str = "paper225") -> np.ndarray:
    """Summarise frame MFCCs as a fixed-length clip vector.

    ``paper225`` resolves an ambiguity in the 2014 paper by using seven base
    statistics plus the mean first and second frame derivatives: 25 x 9 = 225.
    ``literal275`` follows the literal wording by adding both mean and variance
    for each derivative: 25 x 11 = 275. Reporting both makes the reproduction
    transparent instead of silently changing the published dimensionality.
    """
    if matrix.ndim != 2:
        raise ValueError("MFCC matrix must have shape (coefficients, frames).")

    delta = np.diff(matrix, axis=1)
    delta2 = np.diff(delta, axis=1)
    if delta.shape[1] == 0:
        delta = np.zeros((matrix.shape[0], 1), dtype=np.float64)
    if delta2.shape[1] == 0:
        delta2 = np.zeros((matrix.shape[0], 1), dtype=np.float64)

    statistics = _base_statistics(matrix)
    if variant == "paper225":
        statistics.extend([np.mean(delta, axis=1), np.mean(delta2, axis=1)])
    elif variant == "literal275":
        statistics.extend(
            [
                np.mean(delta, axis=1),
                np.var(delta, axis=1),
                np.mean(delta2, axis=1),
                np.var(delta2, axis=1),
            ]
        )
    else:
        raise ValueError(f"Unknown feature variant: {variant}")

    vector = np.concatenate(statistics)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_feature(audio: np.ndarray, config: FeatureConfig) -> np.ndarray:
    return summarise_mfcc(mfcc_matrix(audio, config), config.feature_variant)


def expected_feature_dimension(config: FeatureConfig) -> int:
    multiplier = 9 if config.feature_variant == "paper225" else 11
    return config.n_mfcc * multiplier


def feature_manifest(config: FeatureConfig) -> dict[str, object]:
    manifest = asdict(config)
    manifest["backend"] = "librosa"
    manifest["dimension"] = expected_feature_dimension(config)
    manifest["paper_implementation_note"] = (
        "The 2014 paper used Essentia and reports 225 dimensions although its "
        "wording can be read as 275. paper225 uses nine summaries per MFCC."
    )
    return manifest
