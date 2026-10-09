from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FeatureConfig, load_config
from .dataset import Clip, UrbanSound8K
from .evaluation import fold_metrics
from .feature_store import FeatureSet, cached_feature_set, stable_seed
from .model import make_svm_pipeline
from .noise import EnvironmentalNoise, mix_at_snr, noise_for_condition


def _condition_label(strategy: str, test_condition: str) -> str:
    if test_condition == "original":
        return "original"
    if test_condition == "crowd":
        return "held_out"
    if strategy == test_condition:
        return "matched"
    if strategy == "mixed" and test_condition in {"white", "pink", "traffic"}:
        return "seen_in_mixture"
    return "mismatched"


def _load_noise_banks(
    feature_config: FeatureConfig,
    raw_config: dict[str, object],
    traffic_path: str | Path,
    crowd_path: str | Path,
) -> tuple[EnvironmentalNoise, EnvironmentalNoise]:
    noise_config = raw_config.get("noise", {})
    train_fraction = float(noise_config.get("train_fraction", 0.60))
    guard_fraction = float(noise_config.get("guard_fraction", 0.10))
    traffic = EnvironmentalNoise.from_file(
        traffic_path,
        feature_config.sample_rate,
        str(noise_config.get("traffic_environment", "STRAFFIC")),
        train_fraction,
        guard_fraction,
    )
    crowd = EnvironmentalNoise.from_file(
        crowd_path,
        feature_config.sample_rate,
        str(noise_config.get("crowd_environment", "PCAFETER")),
        train_fraction,
        guard_fraction,
    )
    return traffic, crowd


def _noise_transform(
    condition: str,
    snr_db: float,
    partition: str,
    seed: int,
    traffic: EnvironmentalNoise,
    crowd: EnvironmentalNoise,
):
    def transform(audio: np.ndarray, clip: Clip) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(seed, clip.clip_id, condition, snr_db, partition))
        noise, _ = noise_for_condition(
            condition,
            len(audio),
            rng,
            partition,
            traffic=traffic,
            crowd=crowd,
        )
        mixture, _ = mix_at_snr(audio, noise, snr_db)
        return mixture

    return transform


def _mixed_training_transform(
    conditions: list[str],
    snrs: list[float],
    seed: int,
    traffic: EnvironmentalNoise,
    crowd: EnvironmentalNoise,
):
    def transform(audio: np.ndarray, clip: Clip) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(seed, clip.clip_id, "mixed", "train"))
        condition = str(rng.choice(conditions))
        snr_db = float(rng.choice(snrs))
        noise, _ = noise_for_condition(
            condition,
            len(audio),
            rng,
            "train",
            traffic=traffic,
            crowd=crowd,
        )
        mixture, _ = mix_at_snr(audio, noise, snr_db)
        return mixture

    return transform


def run_experiment(
    dataset_root: str | Path,
    traffic_noise: str | Path,
    crowd_noise: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    n_jobs: int = 1,
    limit: int | None = None,
    folds: list[int] | None = None,
    overwrite_cache: bool = False,
) -> dict[str, object]:
    raw_config = load_config(config_path)
    feature_config = FeatureConfig.from_mapping(raw_config)
    seed = int(raw_config.get("seed", 540334953))
    dataset = UrbanSound8K(dataset_root)
    clips = dataset.clips(limit=limit)
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traffic, crowd = _load_noise_banks(
        feature_config, raw_config, traffic_noise, crowd_noise
    )
    original = cached_feature_set(
        cache_dir / f"original_{feature_config.feature_variant}.npz",
        clips,
        feature_config,
        n_jobs=n_jobs,
        extra_manifest={
            "condition": "original",
            "clip_count": len(clips),
            "dataset_root": str(Path(dataset_root).resolve()),
        },
        overwrite=overwrite_cache,
    )

    training_config = raw_config.get("training", {})
    testing_config = raw_config.get("testing", {})
    strategies = list(training_config.get("strategies", ["original", "white", "traffic", "mixed"]))
    train_snrs = [float(value) for value in training_config.get("augmentation_snrs_db", [0, 10, 20])]
    test_conditions = list(testing_config.get("conditions", ["original", "white", "pink", "traffic", "crowd"]))
    test_snrs = [float(value) for value in testing_config.get("snrs_db", [20, 10, 0])]
    c = float(training_config.get("svm_c", 1.0))
    gamma = training_config.get("svm_gamma", "scale")

    augmented_training: dict[str, FeatureSet] = {}
    for strategy in strategies:
        if strategy == "original":
            continue
        if strategy == "mixed":
            transform = _mixed_training_transform(
                ["white", "pink", "traffic"], train_snrs, seed, traffic, crowd
            )
        else:
            def transform(audio: np.ndarray, clip: Clip, strategy=strategy) -> np.ndarray:
                rng = np.random.default_rng(stable_seed(seed, clip.clip_id, strategy, "train"))
                snr_db = float(rng.choice(train_snrs))
                noise, _ = noise_for_condition(
                    strategy, len(audio), rng, "train", traffic=traffic, crowd=crowd
                )
                return mix_at_snr(audio, noise, snr_db)[0]

        augmented_training[strategy] = cached_feature_set(
            cache_dir / f"train_{strategy}_{feature_config.feature_variant}.npz",
            clips,
            feature_config,
            transform=transform,
            n_jobs=n_jobs,
            extra_manifest={
                "condition": strategy,
                "partition": "train",
                "snrs_db": train_snrs,
                "seed": seed,
                "clip_count": len(clips),
                "dataset_root": str(Path(dataset_root).resolve()),
            },
            overwrite=overwrite_cache,
        )

    tests: dict[tuple[str, float | None], FeatureSet] = {("original", None): original}
    for condition in test_conditions:
        if condition == "original":
            continue
        for snr_db in test_snrs:
            tests[(condition, snr_db)] = cached_feature_set(
                cache_dir / f"test_{condition}_{snr_db:g}db_{feature_config.feature_variant}.npz",
                clips,
                feature_config,
                transform=_noise_transform(condition, snr_db, "test", seed, traffic, crowd),
                n_jobs=n_jobs,
                extra_manifest={
                    "condition": condition,
                    "partition": "test",
                    "snr_db": snr_db,
                    "seed": seed,
                    "clip_count": len(clips),
                    "dataset_root": str(Path(dataset_root).resolve()),
                },
                overwrite=overwrite_cache,
            )

    selected_folds = folds or sorted(map(int, np.unique(original.folds)))
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for strategy in strategies:
        for test_fold in selected_folds:
            train_mask = original.folds != test_fold
            train_x = original.x[train_mask]
            train_y = original.y[train_mask]
            if strategy != "original":
                augmented = augmented_training[strategy]
                train_x = np.vstack([train_x, augmented.x[train_mask]])
                train_y = np.concatenate([train_y, augmented.y[train_mask]])

            model = make_svm_pipeline(c=c, gamma=gamma)
            model.fit(train_x, train_y)
            model_dir = output_dir / "models" / strategy
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"fold{test_fold}.joblib")

            for (condition, snr_db), test_set in tests.items():
                test_mask = test_set.folds == test_fold
                predictions = model.predict(test_set.x[test_mask])
                metrics = fold_metrics(test_set.y[test_mask], predictions)
                metric_rows.append(
                    {
                        "strategy": strategy,
                        "fold": int(test_fold),
                        "test_condition": condition,
                        "snr_db": np.nan if snr_db is None else float(snr_db),
                        "relationship": _condition_label(strategy, condition),
                        **metrics,
                    }
                )
                indices = np.flatnonzero(test_mask)
                for index, predicted in zip(indices, predictions):
                    prediction_rows.append(
                        {
                            "strategy": strategy,
                            "fold": int(test_fold),
                            "test_condition": condition,
                            "snr_db": np.nan if snr_db is None else float(snr_db),
                            "relationship": _condition_label(strategy, condition),
                            "clip_id": str(test_set.clip_ids[index]),
                            "true_class_id": int(test_set.y[index]),
                            "true_class": str(test_set.class_names[index]),
                            "predicted_class_id": int(predicted),
                        }
                    )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    metrics_frame.to_csv(output_dir / "all_fold_metrics.csv", index=False)
    predictions_frame.to_csv(output_dir / "all_predictions.csv", index=False)
    aggregate = (
        metrics_frame.groupby(
            ["strategy", "test_condition", "snr_db", "relationship"], dropna=False
        )[["accuracy", "macro_f1", "macro_recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = ["_".join(filter(None, map(str, item))) for item in aggregate.columns]
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    manifest = {
        "experiment": "matched_mismatched_noise_augmentation",
        "seed": seed,
        "evaluated_folds": selected_folds,
        "training_strategies": strategies,
        "training_snrs_db": train_snrs,
        "testing_conditions": test_conditions,
        "testing_snrs_db": test_snrs,
        "traffic_environment": traffic.environment,
        "crowd_environment_held_out": crowd.environment,
        "noise_partition": {
            "train_fraction": traffic.train_fraction,
            "guard_fraction": traffic.guard_fraction,
            "test_fraction": 1.0 - traffic.train_fraction - traffic.guard_fraction,
        },
        "augmentation_policy": "Each augmented strategy uses original training clips plus one deterministic augmented copy per clip.",
        "data_leakage_control": "Official UrbanSound8K folds; model preprocessing fitted on training folds; disjoint DEMAND time regions; crowd environment excluded from training.",
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched/mismatched noise experiments.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--traffic-noise", required=True, help="DEMAND STRAFFIC WAV file.")
    parser.add_argument("--crowd-noise", required=True, help="DEMAND PCAFETER WAV file.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/full_experiment")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--folds", help="Comma-separated fold numbers.")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    folds = None if not args.folds else [int(item) for item in args.folds.split(",")]
    manifest = run_experiment(
        args.dataset_root,
        args.traffic_noise,
        args.crowd_noise,
        args.config,
        args.output_dir,
        args.cache_dir,
        n_jobs=args.n_jobs,
        limit=args.limit,
        folds=folds,
        overwrite_cache=args.overwrite_cache,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
