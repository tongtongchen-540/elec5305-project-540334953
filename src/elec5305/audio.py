from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load an audio file as finite mono float32 samples at a fixed rate."""
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    return np.nan_to_num(audio, copy=False)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def peak_safe(audio: np.ndarray, peak: float = 0.999) -> tuple[np.ndarray, float]:
    """Apply one common gain only when necessary; return audio and gain."""
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    gain = 1.0 if maximum <= peak or maximum == 0.0 else peak / maximum
    return np.asarray(audio * gain, dtype=np.float32), gain
