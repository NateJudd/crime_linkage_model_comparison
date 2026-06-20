from linkage_compare.compare import compare_models
from linkage_compare.simulate import flatten_linkages, simulate_features, simulate_linkages


def test_compare_models_returns_metrics_and_significance():
    n, p = 60, 6
    linkages = simulate_linkages(n, random_state=1)
    features = simulate_features(n, p, random_state=1)
    y = flatten_linkages(linkages)

    glm_metrics, rf_metrics, significance = compare_models(y, features, random_state=1)

    for metrics in (glm_metrics, rf_metrics):
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    assert 0.0 <= significance.p_value <= 1.0
