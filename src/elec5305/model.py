from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Training-only, documented approximation to Weka CFS.

    Features are ranked by ANOVA class relevance. A candidate is retained if
    its absolute Pearson correlation with every already-retained feature is
    below ``max_correlation``. This is not claimed to be bit-identical to Weka's
    CfsSubsetEval/BestFirst implementation.
    """

    def __init__(self, max_correlation: float = 0.95, max_features: int | None = None):
        self.max_correlation = max_correlation
        self.max_features = max_features

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        scores, _ = f_classif(x, y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        order = np.argsort(scores)[::-1]
        correlation = np.corrcoef(x, rowvar=False)
        correlation = np.nan_to_num(np.abs(correlation), nan=0.0)

        selected: list[int] = []
        for index in order:
            if selected and np.max(correlation[index, selected]) >= self.max_correlation:
                continue
            selected.append(int(index))
            if self.max_features is not None and len(selected) >= self.max_features:
                break
        self.selected_indices_ = np.asarray(sorted(selected), dtype=int)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[:, self.selected_indices_]

    def get_support(self) -> np.ndarray:
        return self.selected_indices_


def make_svm_pipeline(
    c: float = 1.0,
    gamma: str | float = "scale",
    use_correlation_pruner: bool = True,
) -> Pipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if use_correlation_pruner:
        steps.append(("selector", CorrelationPruner()))
    steps.append(
        (
            "svm",
            SVC(
                C=c,
                gamma=gamma,
                kernel="rbf",
                class_weight=None,
            ),
        )
    )
    return Pipeline(steps)
