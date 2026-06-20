"""Model-fitting utilities for the linkage classification task."""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def fit_glm(
    linkages: np.ndarray, features: np.ndarray, random_state: int | None = None
) -> ClassifierMixin:
    """Fit a logistic regression (GLM) baseline model."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(features, linkages)
    return model


def fit_random_forest(
    linkages: np.ndarray, features: np.ndarray, random_state: int | None = None
) -> ClassifierMixin:
    """Fit a random forest (proposed) model."""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(features, linkages)
    return model
