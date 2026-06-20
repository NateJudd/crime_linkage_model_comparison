import numpy as np

from linkage_compare.simulate import flatten_linkages, simulate_features, simulate_linkages


def test_simulate_linkages_shape_and_diagonal():
    linkages = simulate_linkages(10, random_state=0)
    assert linkages.shape == (10, 10)
    assert np.all(np.diag(linkages) == 0)
    assert set(np.unique(linkages)).issubset({0, 1})


def test_simulate_features_shape():
    features = simulate_features(10, 5, random_state=0)
    assert features.shape == (100, 5)


def test_flatten_linkages_matches_features_rows():
    linkages = simulate_linkages(10, random_state=0)
    features = simulate_features(10, 5, random_state=0)
    y = flatten_linkages(linkages)
    assert y.shape[0] == features.shape[0]


def test_reproducible_with_seed():
    a = simulate_linkages(10, random_state=42)
    b = simulate_linkages(10, random_state=42)
    np.testing.assert_array_equal(a, b)
