from __future__ import annotations

from enum import StrEnum, auto
from typing import Callable

import numpy as np


class Metric(StrEnum):
    ACCURACY = auto()
    F1 = auto()


class MetricException(Exception): ...


MetricFunc = Callable[[np.ndarray, np.ndarray], float]


# x are predictions, y are labels
_correct: MetricFunc = lambda x, y: (x == y).sum()

# binary classifications with 0s (neg) and 1s (pos)
_tp: MetricFunc = lambda x, y: ((x == 1) & (y == 1)).sum()  # true positive
_fp: MetricFunc = lambda x, y: ((x == 1) & (y == 0)).sum()  # false positive
_tn: MetricFunc = lambda x, y: ((x == 1) & (y == 1)).sum()  # true negative
_fn: MetricFunc = lambda x, y: ((x == 0) & (y == 1)).sum()  # false negative


def _safe_divide(x: float, y: float) -> float:
    return 0 if y == 0 else x / y


def accuracy_score(preds: np.ndarray, labels: np.ndarray) -> float:
    return _correct(preds, labels) / len(preds)


def precision_score(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = _tp(preds, labels)
    fp = _fp(preds, labels)

    return _safe_divide(tp, tp + fp)


def recall_score(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = _tp(preds, labels)
    fn = _fn(preds, labels)

    return _safe_divide(tp, tp + fn)


def f1_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    F1 score for binary classification
    """
    p = precision_score(preds, labels)
    r = recall_score(preds, labels)

    return _safe_divide(2 * p * r, p + r)  # harmonic mean


METRIC_TO_FUNC: dict[Metric, MetricFunc] = {
    Metric.ACCURACY: accuracy_score,
    Metric.F1: f1_score,
}


def compute_eval_metrics(
    metrics: list[Metric],
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute provided metrics sequentially
    """
    if len(preds) != len(labels):
        raise MetricException(
            "Length of predictions has to be equal to the length of labels,"
            f"currently {len(preds)} vs {len(labels)}"
        )

    results = {}
    for metric in metrics:
        metric_impl = METRIC_TO_FUNC[metric]
        result = metric_impl(preds, labels)
        result = round(100 * result, 2)  # formatting
        results[metric.value] = result

    return results
