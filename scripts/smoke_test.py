from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from elec5305.analysis import run_analysis
from elec5305.baseline import run_baseline
from elec5305.experiment import run_experiment
from elec5305.plot_results import (
    plot_accuracy_curves,
    plot_selected_confusions,
    plot_transfer_heatmap,
)


def make_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    if root.exists():
        shutil.rmtree(root)
    dataset_root = root / "UrbanSound8K"
    (dataset_root / "metadata").mkdir(parents=True)
    sample_rate = 8_000
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(540334953)
    frequencies = [250, 700, 1_500]
    class_names = ["low_tone", "mid_tone", "high_tone"]
    for fold in [1, 2, 3]:
        fold_dir = dataset_root / "audio" / f"fold{fold}"
        fold_dir.mkdir(parents=True)
        for class_id, (frequency, class_name) in enumerate(zip(frequencies, class_names)):
            for example in range(2):
                duration = 0.55 + 0.03 * example
                time = np.arange(int(sample_rate * duration)) / sample_rate
                audio = 0.20 * np.sin(2 * np.pi * frequency * time)
                audio += 0.01 * rng.standard_normal(len(time))
                file_name = f"fold{fold}_class{class_id}_{example}.wav"
                sf.write(fold_dir / file_name, audio.astype(np.float32), sample_rate)
                rows.append(
                    {
                        "slice_file_name": file_name,
                        "fsID": fold * 100 + class_id,
                        "start": 0.0,
                        "end": duration,
                        "salience": 1,
                        "fold": fold,
                        "classID": class_id,
                        "class": class_name,
                    }
                )
    pd.DataFrame(rows).to_csv(dataset_root / "metadata" / "UrbanSound8K.csv", index=False)

    noise_dir = root / "noise"
    noise_dir.mkdir()
    seconds = 12
    time = np.arange(sample_rate * seconds) / sample_rate
    traffic = 0.15 * rng.standard_normal(len(time)) + 0.08 * np.sin(2 * np.pi * 90 * time)
    crowd = 0.12 * rng.standard_normal(len(time)) + 0.04 * np.sin(2 * np.pi * 1_100 * time)
    traffic_path = noise_dir / "STRAFFIC.wav"
    crowd_path = noise_dir / "PCAFETER.wav"
    sf.write(traffic_path, traffic.astype(np.float32), sample_rate)
    sf.write(crowd_path, crowd.astype(np.float32), sample_rate)

    config = {
        "seed": 540334953,
        "sample_rate": sample_rate,
        "frame_length": 256,
        "hop_length": 128,
        "n_fft": 256,
        "n_mels": 40,
        "n_mfcc": 25,
        "fmin": 0.0,
        "fmax": 4_000.0,
        "feature_variant": "paper225",
        "training": {
            "strategies": ["original", "white", "traffic", "mixed"],
            "augmentation_snrs_db": [0, 10, 20],
            "svm_c": 1.0,
            "svm_gamma": "scale",
        },
        "testing": {
            "conditions": ["original", "white", "pink", "traffic", "crowd"],
            "snrs_db": [20, 10, 0],
        },
        "noise": {
            "traffic_environment": "STRAFFIC",
            "crowd_environment": "PCAFETER",
            "train_fraction": 0.60,
            "guard_fraction": 0.10,
        },
    }
    config_path = root / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return dataset_root, traffic_path, crowd_path, config_path


def main() -> None:
    root = Path("tmp/smoke").resolve()
    dataset_root, traffic, crowd, config = make_fixture(root)
    baseline_dir = root / "results" / "baseline"
    full_dir = root / "results" / "full_experiment"
    cache_dir = root / "cache"

    baseline_manifest = run_baseline(
        dataset_root,
        config,
        baseline_dir,
        cache_dir,
        n_jobs=2,
        folds=[1, 2, 3],
        overwrite_cache=True,
    )
    run_experiment(
        dataset_root,
        traffic,
        crowd,
        config,
        full_dir,
        cache_dir,
        n_jobs=2,
        folds=[1, 2, 3],
        overwrite_cache=True,
    )
    representative = dataset_root / "audio" / "fold1" / "fold1_class1_0.wav"
    run_analysis(
        representative,
        "synthetic mid-tone smoke-test clip",
        traffic,
        crowd,
        config,
        root / "results" / "signal_analysis",
    )

    metrics = pd.read_csv(full_dir / "all_fold_metrics.csv")
    predictions = pd.read_csv(full_dir / "all_predictions.csv")
    plot_accuracy_curves(metrics, full_dir / "accuracy_vs_snr.png")
    plot_transfer_heatmap(metrics, full_dir / "transfer_heatmap_0db.png")
    plot_selected_confusions(predictions, full_dir)

    required = [
        baseline_dir / "fold_metrics.csv",
        baseline_dir / "baseline_confusion_matrix.png",
        full_dir / "aggregate_metrics.csv",
        full_dir / "accuracy_vs_snr.png",
        full_dir / "transfer_heatmap_0db.png",
        root / "results" / "signal_analysis" / "representation_distances.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Smoke test did not create: {missing}")
    print("Smoke test completed successfully.")
    print("Baseline summary:", baseline_manifest["summary"])


if __name__ == "__main__":
    main()
