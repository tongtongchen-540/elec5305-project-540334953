from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_audio, peak_safe


EPSILON = 1e-12


def signal_power(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float64)
    return float(np.mean(np.square(audio))) if audio.size else 0.0


def measured_snr_db(signal: np.ndarray, added_noise: np.ndarray) -> float:
    return 10.0 * np.log10((signal_power(signal) + EPSILON) / (signal_power(added_noise) + EPSILON))


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    prevent_clipping: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mix equal-length arrays at an RMS power SNR.

    Signal and noise power are measured over the complete clip after DC removal.
    If required, one common gain is applied to the mixture. This avoids nonlinear
    clipping and leaves the intended signal-to-noise ratio unchanged.
    """
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if signal.shape != noise.shape:
        raise ValueError("Signal and noise must have identical shapes.")

    signal_dc = signal - np.mean(signal)
    noise_dc = noise - np.mean(noise)
    p_signal = signal_power(signal_dc)
    p_noise = signal_power(noise_dc)
    if p_signal <= EPSILON:
        raise ValueError("Signal power is zero; SNR is undefined.")
    if p_noise <= EPSILON:
        raise ValueError("Noise power is zero; SNR is undefined.")

    scale = np.sqrt(p_signal / (p_noise * (10.0 ** (snr_db / 10.0))))
    scaled_noise = noise_dc * scale
    mixture = signal + scaled_noise
    common_gain = 1.0
    if prevent_clipping:
        mixture, common_gain = peak_safe(mixture)

    achieved = measured_snr_db(signal_dc * common_gain, scaled_noise * common_gain)
    metadata = {
        "target_snr_db": float(snr_db),
        "achieved_snr_db": float(achieved),
        "noise_scale": float(scale),
        "common_gain": float(common_gain),
        "signal_power": float(p_signal),
        "raw_noise_power": float(p_noise),
    }
    return np.asarray(mixture, dtype=np.float32), metadata


def white_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(length).astype(np.float32)


def pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate deterministic FFT-shaped 1/f pink noise."""
    frequencies = np.fft.rfftfreq(length)
    spectrum = rng.standard_normal(frequencies.size) + 1j * rng.standard_normal(frequencies.size)
    weights = np.zeros_like(frequencies)
    weights[1:] = 1.0 / np.sqrt(frequencies[1:])
    shaped = spectrum * weights
    audio = np.fft.irfft(shaped, n=length)
    audio -= np.mean(audio)
    std = np.std(audio)
    if std <= EPSILON:
        raise ValueError("Failed to generate non-zero pink noise.")
    return np.asarray(audio / std, dtype=np.float32)


@dataclass
class EnvironmentalNoise:
    audio: np.ndarray
    sample_rate: int
    environment: str
    source_path: str
    train_fraction: float = 0.60
    guard_fraction: float = 0.10

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        sample_rate: int,
        environment: str,
        train_fraction: float = 0.60,
        guard_fraction: float = 0.10,
    ) -> "EnvironmentalNoise":
        return cls(
            audio=load_audio(path, sample_rate),
            sample_rate=sample_rate,
            environment=environment,
            source_path=str(Path(path)),
            train_fraction=train_fraction,
            guard_fraction=guard_fraction,
        )

    def sample(self, length: int, partition: str, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        total = len(self.audio)
        train_end = int(total * self.train_fraction)
        test_start = int(total * (self.train_fraction + self.guard_fraction))
        if partition == "train":
            low, high = 0, train_end
        elif partition == "test":
            low, high = test_start, total
        else:
            raise ValueError("partition must be 'train' or 'test'.")
        if high - low < length:
            raise ValueError(
                f"{self.environment} {partition} partition is shorter than the requested clip."
            )
        start = int(rng.integers(low, high - length + 1))
        return np.asarray(self.audio[start : start + length], dtype=np.float32), start


def noise_for_condition(
    condition: str,
    length: int,
    rng: np.random.Generator,
    partition: str,
    traffic: EnvironmentalNoise | None = None,
    crowd: EnvironmentalNoise | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if condition == "white":
        return white_noise(length, rng), {"noise_condition": "white", "synthetic": True}
    if condition == "pink":
        return pink_noise(length, rng), {"noise_condition": "pink", "synthetic": True}
    if condition in {"traffic", "crowd"}:
        bank = traffic if condition == "traffic" else crowd
        if bank is None:
            raise ValueError(f"No environmental noise configured for {condition}.")
        excerpt, start = bank.sample(length, partition, rng)
        return excerpt, {
            "noise_condition": condition,
            "synthetic": False,
            "environment": bank.environment,
            "source_path": bank.source_path,
            "partition": partition,
            "start_sample": start,
        }
    raise ValueError(f"Unknown noise condition: {condition}")
