"""Statistical comparison ("A/B testing") utilities for paired classifiers.

When two models are evaluated on the *same* held-out set, their accuracy
scores aren't independent samples, so a plain t-test on accuracy is the
wrong tool. McNemar's test instead looks only at the examples where the
two models disagree and asks whether one model is wrong on those
disagreements significantly more often than the other -- the standard way
to A/B test two paired classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass
class McNemarResult:
    """Result of a McNemar test comparing two paired classifiers.

    Attributes
    ----------
    n01:
        Number of examples where model A was wrong and model B was right.
    n10:
        Number of examples where model A was right and model B was wrong.
    statistic:
        The smaller of ``n01`` and ``n10`` (the exact-test statistic).
    p_value:
        Two-sided p-value testing whether ``n01`` and ``n10`` differ from
        what's expected if neither model disagrees more often than the
        other (i.e. a binomial test against p=0.5).
    """

    n01: int
    n10: int
    statistic: float
    p_value: float


def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> McNemarResult:
    """Run McNemar's exact test to compare two classifiers on the same test set.

    A small p-value suggests the two models' disagreements are lopsided
    enough that the difference in their error rates is unlikely to be due
    to chance alone.
    """
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n10 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong

    n = n01 + n10
    if n == 0:
        return McNemarResult(n01=n01, n10=n10, statistic=0.0, p_value=1.0)

    result = binomtest(min(n01, n10), n, p=0.5)
    return McNemarResult(
        n01=n01, n10=n10, statistic=float(min(n01, n10)), p_value=result.pvalue
    )
