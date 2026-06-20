# linkage-compare

Simulate synthetic record-linkage data and compare a logistic regression
(GLM) baseline against a random forest classifier, including a statistical
significance test between the two.

## What changed from the original version

This started as a small script with a few bugs; this cleanup fixed:

- **`simulate_features` crashed.** It called `np.random.binomial(size=(n, p))`
  without the required `n` (trials) and `p` (probability) arguments.
- **Shape mismatch between features and labels.** `simulate_linkages`
  produced an `n x n` matrix while `simulate_features` produced an `n x p`
  matrix — these can't be fit together. Pairwise linkage classification
  needs one feature vector *per pair*, so `simulate_features` now returns
  `(n * n) x p` rows, and a new `flatten_linkages` helper flattens the
  linkage matrix to match.
- **Models were evaluated on their own training data**, which especially
  inflates the random forest's apparent performance. `compare_models` now
  does a train/test split and scores on the held-out test set.
- **No way to tell if the difference between models was real.** Added a
  McNemar's test (`stats.py`) — the standard way to "A/B test" two
  classifiers evaluated on the same test set — alongside the metrics.
- Added type hints, docstrings, reproducible RNG seeding, packaging,
  linting, and tests so it's ready to push to GitHub.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
python -m linkage_compare
```

```python
from linkage_compare import (
    compare_models,
    flatten_linkages,
    simulate_features,
    simulate_linkages,
)

linkages = simulate_linkages(n=200, random_state=0)
features = simulate_features(n=200, p=10, random_state=0)
y = flatten_linkages(linkages)

glm_metrics, rf_metrics, significance = compare_models(y, features, random_state=0)
print(glm_metrics, rf_metrics, significance)
```

`significance` is a `McNemarResult` with `n01`, `n10`, and `p_value` —
a small `p_value` means the two models' disagreements are lopsided enough
that the performance difference is unlikely to be due to chance.

## Examples

```bash
pip install -e ".[examples]"
python examples/run_comparison.py
```

Writes figures and text results to `examples/output/`:

```
examples/output/
    figures/
        metric_comparison.png
        roc_curves.png
        confusion_matrices.png
    results/
        metrics.txt
        mcnemar_test.txt
```

Re-running the script overwrites these files in place. With the default
purely-random simulation both models score around 0.5 on every metric —
that's expected, since there's no real signal between the simulated
features and labels. Swap in real features/labels (or adjust `p_link`)
to see the comparison do something more interesting.

## Project structure

```
src/linkage_compare/
    simulate.py   # simulate linkage labels and pairwise features
    models.py     # fit GLM / random forest
    compare.py    # train/test split, metrics, McNemar significance test
    stats.py      # McNemar's test implementation
    __main__.py   # demo entry point: python -m linkage_compare
tests/            # pytest unit tests
examples/         # end-to-end example script + generated figures/results
```

## Development

```bash
ruff check .          # lint
black .               # format
pytest                # run tests
pre-commit install    # optional: run lint/format checks on every commit
```

CI (`.github/workflows/ci.yml`) runs lint, format checks, and tests on
Python 3.10–3.12 for every push and pull request.

## Notes / things to double check before relying on this

- `p_link` and feature `prob` both default to 0.5 in the simulators — these
  are placeholders, not estimated from real data.
- The LICENSE is MIT with a placeholder name; update it (or swap licenses)
  before publishing.
