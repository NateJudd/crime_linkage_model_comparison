# linkage-compare

Simulate synthetic record-linkage data and compare a logistic regression
(GLM) baseline against a random forest classifier, including a statistical
significance test between the two.

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
purely-random simulation both models score around 0.5 on every metric (as
expected). Swap in real features/labels (or adjust `p_link`)
to extend the comparison.

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
