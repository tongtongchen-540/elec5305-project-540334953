from __future__ import annotations

import argparse
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from .audio import load_audio, save_audio
from .config import FeatureConfig, load_config
from .features import log_mel_spectrogram, mfcc_matrix
from .feature_store import stable_seed
from .noise import EnvironmentalNoise, mix_at_snr, noise_for_condition


def relative_frobenius(reference: np.ndarray, changed: np.ndarray) -> float:
    frames = min(reference.shape[1], changed.shape[1])
    reference = reference[:, :frames]
    changed = changed[:, :frames]
    return float(np.linalg.norm(changed - reference) / (np.linalg.norm(reference) + 1e-12))


def _plot_representation_grid(
    versions: list[tuple[str, np.ndarray]],
    feature_config: FeatureConfig,
    output_path: Path,
    class_label: str,
) -> None:
    rows = len(versions)
    figure, axes = plt.subplots(rows, 3, figsize=(13, 2.5 * rows), squeeze=False)
    for row, (label, audio) in enumerate(versions):
        time = np.arange(len(audio)) / feature_config.sample_rate
        axes[row, 0].plot(time, audio, linewidth=0.6, color="#2c7fb8")
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_xlim(0, time[-1])
        if row == rows - 1:
            axes[row, 0].set_xlabel("Time (s)")

        log_mel = log_mel_spectrogram(audio, feature_config)
        librosa.display.specshow(
            log_mel,
            sr=feature_config.sample_rate,
            hop_length=feature_config.hop_length,
            x_axis="time",
            y_axis="mel",
            ax=axes[row, 1],
            cmap="magma",
        )
        if row == rows - 1:
            axes[row, 1].set_xlabel("Time (s)")

        mfcc = mfcc_matrix(audio, feature_config)
        librosa.display.specshow(
            mfcc,
            sr=feature_config.sample_rate,
            hop_length=feature_config.hop_length,
            x_axis="time",
            ax=axes[row, 2],
            cmap="coolwarm",
        )
        axes[row, 2].set_ylabel("MFCC index")
        if row == rows - 1:
            axes[row, 2].set_xlabel("Time (s)")

    axes[0, 0].set_title("Waveform")
    axes[0, 1].set_title("Log-Mel spectrum (dB)")
    axes[0, 2].set_title("MFCC representation")
    figure.suptitle(f"Representation changes for {class_label}", y=1.01, fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_noise_spectra(
    noises: dict[str, np.ndarray], sample_rate: int, output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, noise in noises.items():
        frequency, power = welch(noise, fs=sample_rate, nperseg=min(4096, len(noise)))
        axis.semilogx(frequency[1:], 10 * np.log10(power[1:] + 1e-15), label=name)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density (dB/Hz)")
    axis.set_title("Spectral characteristics of added noise")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_analysis(
    audio_path: str | Path,
    class_label: str,
    traffic_noise_path: str | Path,
    crowd_noise_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    raw_config = load_config(config_path)
    feature_config = FeatureConfig.from_mapping(raw_config)
    seed = int(raw_config.get("seed", 540334953))
    snrs = [float(value) for value in raw_config.get("testing", {}).get("snrs_db", [20, 10, 0])]
    noise_config = raw_config.get("noise", {})
    traffic = EnvironmentalNoise.from_file(
        traffic_noise_path,
        feature_config.sample_rate,
        str(noise_config.get("traffic_environment", "STRAFFIC")),
        float(noise_config.get("train_fraction", 0.60)),
        float(noise_config.get("guard_fraction", 0.10)),
    )
    crowd = EnvironmentalNoise.from_file(
        crowd_noise_path,
        feature_config.sample_rate,
        str(noise_config.get("crowd_environment", "PCAFETER")),
        float(noise_config.get("train_fraction", 0.60)),
        float(noise_config.get("guard_fraction", 0.10)),
    )
    original = load_audio(audio_path, feature_config.sample_rate)
    original_mel = log_mel_spectrogram(original, feature_config)
    original_mfcc = mfcc_matrix(original, feature_config)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    representative_noises: dict[str, np.ndarray] = {}

    for condition in ["white", "pink", "traffic", "crowd"]:
        versions: list[tuple[str, np.ndarray]] = [("Original", original)]
        for snr_db in snrs:
            rng = np.random.default_rng(stable_seed(seed, Path(audio_path).name, condition, snr_db))
            noise, noise_metadata = noise_for_condition(
                condition,
                len(original),
                rng,
                "test",
                traffic=traffic,
                crowd=crowd,
            )
            if condition not in representative_noises:
                representative_noises[condition] = noise
            mixture, mix_metadata = mix_at_snr(original, noise, snr_db)
            versions.append((f"{condition}, {snr_db:g} dB", mixture))
            noisy_mel = log_mel_spectrogram(mixture, feature_config)
            noisy_mfcc = mfcc_matrix(mixture, feature_config)
            rows.append(
                {
                    "noise_condition": condition,
                    "environment": noise_metadata.get("environment", condition),
                    "snr_db": snr_db,
                    "achieved_snr_db": mix_metadata["achieved_snr_db"],
                    "relative_log_mel_distance": relative_frobenius(original_mel, noisy_mel),
                    "relative_mfcc_distance": relative_frobenius(original_mfcc, noisy_mfcc),
                }
            )
            save_audio(
                output_dir / "audio_examples" / f"{condition}_{snr_db:g}db.wav",
                mixture,
                feature_config.sample_rate,
            )
        _plot_representation_grid(
            versions,
            feature_config,
            output_dir / f"representation_{condition}.png",
            class_label,
        )

    _plot_noise_spectra(
        representative_noises,
        feature_config.sample_rate,
        output_dir / "noise_power_spectra.png",
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "representation_distances.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse signal, log-Mel and MFCC changes.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--class-label", default="representative UrbanSound8K clip")
    parser.add_argument("--traffic-noise", required=True)
    parser.add_argument("--crowd-noise", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/signal_analysis")
    args = parser.parse_args()
    frame = run_analysis(
        args.audio,
        args.class_label,
        args.traffic_noise,
        args.crowd_noise,
        args.config,
        args.output_dir,
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
