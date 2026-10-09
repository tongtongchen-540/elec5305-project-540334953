from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    class_names: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids, normalize="true")
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_run_summary(
    output_dir: str | Path,
    metrics: list[dict[str, object]],
    predictions: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    numeric = metrics_frame.select_dtypes(include=["number"])
    summary = {
        column: {
            "mean": float(numeric[column].mean()),
            "std": float(numeric[column].std(ddof=1)),
        }
        for column in numeric.columns
        if column != "fold"
    }
    manifest = dict(manifest)
    manifest["summary"] = summary
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
