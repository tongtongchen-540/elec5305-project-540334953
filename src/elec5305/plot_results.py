from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_accuracy_curves(metrics: pd.DataFrame, output_path: Path) -> None:
    noisy = metrics[metrics["test_condition"] != "original"].copy()
    conditions = [item for item in ["white", "pink", "traffic", "crowd"] if item in set(noisy.test_condition)]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, conditions):
        subset = noisy[noisy.test_condition == condition]
        summary = subset.groupby(["strategy", "snr_db"])["accuracy"].agg(["mean", "std"]).reset_index()
        for strategy, group in summary.groupby("strategy"):
            group = group.sort_values("snr_db")
            axis.errorbar(
                group.snr_db,
                group["mean"],
                yerr=group["std"].fillna(0),
                marker="o",
                capsize=2,
                label=strategy,
            )
        axis.set_title(condition)
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1)
    for axis in axes[-1, :]:
        axis.set_xlabel("SNR (dB)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("Accuracy under matched, mismatched and held-out noise")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_transfer_heatmap(metrics: pd.DataFrame, output_path: Path, snr_db: float = 0.0) -> None:
    subset = metrics[(metrics.test_condition != "original") & (metrics.snr_db == snr_db)]
    table = subset.pivot_table(
        index="strategy", columns="test_condition", values="accuracy", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=axis)
    axis.set_xlabel("Test noise")
    axis.set_ylabel("Training strategy")
    axis.set_title(f"Noise transfer matrix at {snr_db:g} dB SNR")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_selected_confusions(predictions: pd.DataFrame, output_dir: Path) -> None:
    selections = [
        ("original", "original", np.nan),
        ("mixed", "crowd", 0.0),
    ]
    class_table = predictions[["true_class_id", "true_class"]].drop_duplicates().sort_values("true_class_id")
    labels = class_table.true_class_id.astype(int).tolist()
    names = class_table.true_class.astype(str).tolist()
    for strategy, condition, snr in selections:
        mask = (predictions.strategy == strategy) & (predictions.test_condition == condition)
        if np.isnan(snr):
            mask &= predictions.snr_db.isna()
        else:
            mask &= predictions.snr_db == snr
        subset = predictions[mask]
        if subset.empty:
            continue
        matrix = confusion_matrix(
            subset.true_class_id,
            subset.predicted_class_id,
            labels=labels,
            normalize="true",
        )
        figure, axis = plt.subplots(figsize=(9, 7))
        sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, xticklabels=names, yticklabels=names, square=True, ax=axis)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        suffix = "original" if np.isnan(snr) else f"{snr:g} dB"
        axis.set_title(f"{strategy} training, {condition} test ({suffix})")
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_{strategy}_{condition}_{suffix.replace(' ', '_')}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full-experiment result tables.")
    parser.add_argument("--results-dir", default="results/full_experiment")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    metrics = pd.read_csv(results_dir / "all_fold_metrics.csv")
    predictions = pd.read_csv(results_dir / "all_predictions.csv")
    plot_accuracy_curves(metrics, results_dir / "accuracy_vs_snr.png")
    plot_transfer_heatmap(metrics, results_dir / "transfer_heatmap_0db.png")
    plot_selected_confusions(predictions, results_dir)


if __name__ == "__main__":
    main()
