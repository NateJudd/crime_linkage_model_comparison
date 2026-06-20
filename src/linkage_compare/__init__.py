"""linkage_compare: simulate record-linkage data and compare classifiers."""

from .compare import compare_models, score_predictions
from .models import fit_glm, fit_random_forest
from .simulate import flatten_linkages, simulate_features, simulate_linkages
from .stats import McNemarResult, mcnemar_test

__all__ = [
    "compare_models",
    "score_predictions",
    "fit_glm",
    "fit_random_forest",
    "flatten_linkages",
    "simulate_features",
    "simulate_linkages",
    "McNemarResult",
    "mcnemar_test",
]

__version__ = "0.1.0"
