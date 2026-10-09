from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FeatureConfig, load_config
from .dataset import UrbanSound8K
from .evaluation import fold_metrics, save_confusion_matrix, write_run_summary
from .feature_store import cached_feature_set
from .features import feature_manifest
from .model import make_svm_pipeline


def run_baseline(
    dataset_root: str | Path,
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
    dataset = UrbanSound8K(dataset_root)
    clips = dataset.clips(limit=limit)
    cache_path = Path(cache_dir) / f"original_{feature_config.feature_variant}.npz"
    features = cached_feature_set(
        cache_path,
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

    available_folds = sorted(map(int, np.unique(features.folds)))
    selected_folds = folds or available_folds
    training_config = raw_config.get("training", {})
    model_c = float(training_config.get("svm_c", 1.0))
    model_gamma = training_config.get("svm_gamma", "scale")

    all_true: list[int] = []
    all_pred: list[int] = []
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_counts: list[int] = []

    for test_fold in selected_folds:
        test_mask = features.folds == test_fold
        train_mask = ~test_mask
        if not np.any(test_mask) or len(np.unique(features.y[train_mask])) < 2:
            raise ValueError(f"Fold {test_fold} cannot be evaluated with the available clips.")

        model = make_svm_pipeline(c=model_c, gamma=model_gamma)
        model.fit(features.x[train_mask], features.y[train_mask])
        predictions = model.predict(features.x[test_mask])
        row = {"fold": int(test_fold), **fold_metrics(features.y[test_mask], predictions)}
        metric_rows.append(row)
        selected_counts.append(len(model.named_steps["selector"].get_support()))

        indices = np.flatnonzero(test_mask)
        for index, predicted in zip(indices, predictions):
            prediction_rows.append(
                {
                    "clip_id": str(features.clip_ids[index]),
                    "fold": int(test_fold),
                    "true_class_id": int(features.y[index]),
                    "true_class": str(features.class_names[index]),
                    "predicted_class_id": int(predicted),
                }
            )
        all_true.extend(features.y[test_mask].tolist())
        all_pred.extend(predictions.tolist())

    class_mapping = (
        pd.DataFrame({"id": features.y, "name": features.class_names})
        .drop_duplicates()
        .sort_values("id")
    )
    output_dir = Path(output_dir)
    save_confusion_matrix(
        np.asarray(all_true),
        np.asarray(all_pred),
        class_mapping["id"].astype(int).tolist(),
        class_mapping["name"].astype(str).tolist(),
        output_dir / "baseline_confusion_matrix.png",
        "Published-baseline reproduction: normalised confusion matrix",
    )
    manifest = {
        "experiment": "published_baseline_reproduction",
        "dataset_root": str(Path(dataset_root).resolve()),
        "evaluated_folds": selected_folds,
        "feature": feature_manifest(feature_config),
        "classifier": {
            "implementation": "sklearn.svm.SVC",
            "kernel": "rbf",
            "C": model_c,
            "gamma": model_gamma,
            "scaling": "StandardScaler fitted on each training fold",
            "selector": "CorrelationPruner (documented Python approximation to Weka CFS)",
            "selected_feature_count_mean": float(np.mean(selected_counts)),
        },
        "reproduction_limitations": [
            "The paper used Essentia; this implementation uses librosa.",
            "The paper used Weka CFS and an RBF SVM; this implementation uses a documented sklearn approximation.",
            "The paper's statistic wording and reported 225 dimensions are internally ambiguous.",
        ],
    }
    write_run_summary(output_dir, metric_rows, pd.DataFrame(prediction_rows), manifest)
    return json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))


def parse_folds(value: str | None) -> list[int] | None:
    return None if not value else [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the UrbanSound8K MFCC-SVM baseline.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--folds", help="Comma-separated folds; default is all available folds.")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    manifest = run_baseline(
        args.dataset_root,
        args.config,
        args.output_dir,
        args.cache_dir,
        n_jobs=args.n_jobs,
        limit=args.limit,
        folds=parse_folds(args.folds),
        overwrite_cache=args.overwrite_cache,
    )
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
