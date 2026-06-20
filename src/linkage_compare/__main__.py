"""Run a demo comparison of GLM vs random forest on simulated linkage data."""

from __future__ import annotations

from .compare import compare_models
from .simulate import flatten_linkages, simulate_features, simulate_linkages


def main() -> None:
    n, p = 200, 10
    random_state = 0

    linkages = simulate_linkages(n, random_state=random_state)
    features = simulate_features(n, p, random_state=random_state)
    y = flatten_linkages(linkages)

    glm_metrics, rf_metrics, significance = compare_models(y, features, random_state=random_state)

    print("GLM metrics:", glm_metrics)
    print("Random forest metrics:", rf_metrics)
    print(
        f"McNemar test: n01={significance.n01}, n10={significance.n10}, "
        f"p_value={significance.p_value:.4f}"
    )


if __name__ == "__main__":
    main()
