import numpy as np

from linkage_compare.models import fit_glm, fit_random_forest


def _toy_data():
    rng = np.random.default_rng(0)
    X = rng.binomial(1, 0.5, size=(50, 4))
    y = rng.binomial(1, 0.5, size=50)
    return X, y


def test_fit_glm_predicts_binary():
    X, y = _toy_data()
    model = fit_glm(y, X, random_state=0)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_fit_random_forest_predicts_binary():
    X, y = _toy_data()
    model = fit_random_forest(y, X, random_state=0)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
