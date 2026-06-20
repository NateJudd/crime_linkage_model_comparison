"""Simulate synthetic data for the record-linkage classification task.

Every ordered pair of records ``(i, j)`` with ``i != j`` is treated as one
labeled example: ``linkages[i, j] == 1`` means that pair is a true match.
``simulate_features`` produces one feature vector per pair (``n * n`` rows),
and :func:`flatten_linkages` flattens the linkage matrix into a label
vector in the same row order, so the two line up 1:1 for model fitting.
"""

from __future__ import annotations

import numpy as np


def simulate_linkages(
    n: int, p_link: float = 0.5, random_state: int | None = None
) -> np.ndarray:
    """Simulate an n x n binary matrix of pairwise linkages.

    Parameters
    ----------
    n:
        Number of records.
    p_link:
        Probability that any given ordered pair is a true linkage.
    random_state:
        Optional seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    linkages = rng.binomial(1, p_link, size=(n, n))
    np.fill_diagonal(linkages, 0)
    return linkages


def simulate_features(
    n: int, p: int, prob: float = 0.5, random_state: int | None = None
) -> np.ndarray:
    """Simulate an (n * n) x p matrix of binary features, one row per ordered pair.

    Parameters
    ----------
    n:
        Number of records (must match the ``n`` used for ``simulate_linkages``
        so that rows line up with the flattened linkage labels).
    p:
        Number of binary features per pair.
    prob:
        Probability that a given feature is 1.
    random_state:
        Optional seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    return rng.binomial(1, prob, size=(n * n, p))


def flatten_linkages(linkages: np.ndarray) -> np.ndarray:
    """Flatten an n x n linkage matrix into a 1-D label vector.

    Uses row-major ("C") order, matching the row order produced by
    :func:`simulate_features`.
    """
    return linkages.flatten()
