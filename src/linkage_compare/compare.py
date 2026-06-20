"""Compare the performance of the GLM and random forest linkage models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .models import fit_glm, fit_random_forest
from .stats import McNemarResult, mcnemar_test


def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Compute the standard metric suite for a single model's predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def compare_models(
    linkages: np.ndarray,
    features: np.ndarray,
    test_size: float = 0.3,
    random_state: int | None = None,
) -> tuple[dict[str, float], dict[str, float], McNemarResult]:
    """Fit GLM and random forest models and compare their held-out performance.

    Models are trained on a split of the data and scored on a held-out
    test split, so the metrics reflect generalization rather than fit to
    the training data (a random forest in particular can otherwise
    memorize the training set and look artificially strong). A McNemar
    test is also run on the two models' test-set predictions to check
    whether the difference between them is statistically significant.

    Returns
    -------
    A tuple of ``(glm_metrics, rf_metrics, significance)``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        linkages,
        test_size=test_size,
        random_state=random_state,
        stratify=linkages,
    )

    glm_model = fit_glm(y_train, X_train, random_state=random_state)
    rf_model = fit_random_forest(y_train, X_train, random_state=random_state)

    y_pred_glm = glm_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_score_glm = glm_model.predict_proba(X_test)[:, 1]
    y_score_rf = rf_model.predict_proba(X_test)[:, 1]

    glm_metrics = score_predictions(y_test, y_pred_glm, y_score_glm)
    rf_metrics = score_predictions(y_test, y_pred_rf, y_score_rf)
    significance = mcnemar_test(y_test, y_pred_glm, y_pred_rf)

    return glm_metrics, rf_metrics, significance
