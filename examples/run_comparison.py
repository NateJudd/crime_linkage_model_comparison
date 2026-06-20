"""End-to-end example: simulate data, compare models, and save outputs.

Run from the repo root with:

    python examples/run_comparison.py

Writes:
    examples/output/results/metrics.txt       -- formatted metric table
    examples/output/results/mcnemar_test.txt  -- significance test + interpretation
    examples/output/figures/metric_comparison.png
    examples/output/figures/roc_curves.png
    examples/output/figures/confusion_matrices.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from linkage_compare import (
    fit_glm,
    fit_random_forest,
    flatten_linkages,
    mcnemar_test,
    score_predictions,
    simulate_features,
    simulate_linkages,
)

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

N_RECORDS = 200
N_FEATURES = 10
TEST_SIZE = 0.3
RANDOM_STATE = 0


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- simulate + split -------------------------------------------------
    linkages = simulate_linkages(N_RECORDS, random_state=RANDOM_STATE)
    features = simulate_features(N_RECORDS, N_FEATURES, random_state=RANDOM_STATE)
    y = flatten_linkages(linkages)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- fit models ---------------------------------------------------------
    glm_model = fit_glm(y_train, X_train, random_state=RANDOM_STATE)
    rf_model = fit_random_forest(y_train, X_train, random_state=RANDOM_STATE)

    y_pred_glm = glm_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_score_glm = glm_model.predict_proba(X_test)[:, 1]
    y_score_rf = rf_model.predict_proba(X_test)[:, 1]

    glm_metrics = score_predictions(y_test, y_pred_glm, y_score_glm)
    rf_metrics = score_predictions(y_test, y_pred_rf, y_score_rf)
    significance = mcnemar_test(y_test, y_pred_glm, y_pred_rf)

    # --- write results/metrics.txt ------------------------------------------
    metrics_path = RESULTS_DIR / "metrics.txt"
    with metrics_path.open("w") as f:
        f.write(f"Simulation: n={N_RECORDS} records, p={N_FEATURES} features, ")
        f.write(f"test_size={TEST_SIZE}, random_state={RANDOM_STATE}\n")
        f.write(f"Test set size: {len(y_test)} pairs\n\n")
        header = f"{'metric':<10}{'GLM':>10}{'RandomForest':>15}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            f.write(f"{key:<10}{glm_metrics[key]:>10.4f}{rf_metrics[key]:>15.4f}\n")
    print(f"Wrote {metrics_path}")

    # --- write results/mcnemar_test.txt -------------------------------------
    mcnemar_path = RESULTS_DIR / "mcnemar_test.txt"
    with mcnemar_path.open("w") as f:
        f.write("McNemar's test: GLM vs RandomForest on the same test set\n")
        f.write("-" * 60 + "\n")
        f.write(f"n01 (GLM wrong, RF right): {significance.n01}\n")
        f.write(f"n10 (GLM right, RF wrong): {significance.n10}\n")
        f.write(f"p_value: {significance.p_value:.4f}\n\n")
        if significance.p_value < 0.05:
            f.write(
                "p < 0.05: the two models' disagreements are lopsided enough that "
                "the performance difference is unlikely to be due to chance.\n"
            )
        else:
            f.write(
                "p >= 0.05: not enough evidence to say the two models perform "
                "differently on this test set.\n"
            )
    print(f"Wrote {mcnemar_path}")

    # --- figure: metric comparison bar chart --------------------------------
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    glm_values = [glm_metrics[m] for m in metric_names]
    rf_values = [rf_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, glm_values, width, label="GLM")
    ax.bar(x + width / 2, rf_values, width, label="Random Forest")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("GLM vs Random Forest: held-out test metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metric_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'metric_comparison.png'}")

    # --- figure: ROC curves --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, y_score in (("GLM", y_score_glm), ("Random Forest", y_score_rf)):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (held-out test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'roc_curves.png'}")

    # --- figure: confusion matrices -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, label, y_pred in (
        (axes[0], "GLM", y_pred_glm),
        (axes[1], "Random Forest", y_pred_rf),
    ):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["no link", "link"]).plot(ax=ax, colorbar=False)
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURES_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
