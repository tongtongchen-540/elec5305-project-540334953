from __future__ import annotations

import numpy as np

from elec5305.config import FeatureConfig
from elec5305.features import extract_feature, mfcc_matrix, summarise_mfcc
from elec5305.noise import measured_snr_db, mix_at_snr, pink_noise, white_noise


def test_published_feature_dimension() -> None:
    config = FeatureConfig(sample_rate=8_000, frame_length=256, hop_length=128, n_fft=256, fmax=4_000)
    time = np.arange(8_000) / 8_000
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    vector = extract_feature(audio, config)
    assert vector.shape == (225,)
    assert np.all(np.isfinite(vector))


def test_literal_interpretation_dimension() -> None:
    matrix = np.arange(25 * 20, dtype=float).reshape(25, 20)
    assert summarise_mfcc(matrix, "literal275").shape == (275,)


def test_snr_mixing_accuracy() -> None:
    rng = np.random.default_rng(123)
    signal = rng.standard_normal(16_000).astype(np.float32) * 0.05
    noise = white_noise(len(signal), rng)
    _, metadata = mix_at_snr(signal, noise, 10.0)
    assert abs(metadata["achieved_snr_db"] - 10.0) < 1e-6


def test_noise_generators_are_finite() -> None:
    rng = np.random.default_rng(456)
    assert np.all(np.isfinite(white_noise(4_096, rng)))
    assert np.all(np.isfinite(pink_noise(4_096, rng)))


def test_mfcc_matrix_shape() -> None:
    config = FeatureConfig(sample_rate=8_000, frame_length=256, hop_length=128, n_fft=256, fmax=4_000)
    matrix = mfcc_matrix(np.ones(2_000, dtype=np.float32), config)
    assert matrix.shape[0] == 25
    assert matrix.shape[1] > 1
