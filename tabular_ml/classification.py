"""Breast cancer diagnosis: Logistic Regression, Decision Tree, KNN and Naive Bayes with grid search."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from tabular_ml.data import RESULTS_DIR, load_breast_cancer_df
from tabular_ml.plotting import heatmap, plt, save

SEED = 42
CV_FOLDS = 5
LABELS = ["Malignant", "Benign"]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clips each feature to quantiles learned on the training data."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def build_models() -> dict:
    return {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
            {"clf__max_depth": [5, 10, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "KNN": (make_pipeline(KNeighborsClassifier()), {"clf__n_neighbors": [3, 5, 7, 11]}),
        "Gaussian NB": (make_pipeline(GaussianNB()), None),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, models: dict | None = None) -> list[dict]:
    results = []
    for name, (pipe, grid) in (models or build_models()).items():
        if grid:
            search = GridSearchCV(pipe, grid, cv=CV_FOLDS, scoring="f1", n_jobs=-1).fit(X_train, y_train)
            model, params = search.best_estimator_, search.best_params_
        else:
            model, params = pipe.fit(X_train, y_train), {}
        cv = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "model": name,
            "best_params": {k.removeprefix("clf__"): v for k, v in params.items()},
            "cv_f1_mean": cv.mean(),
            "cv_f1_std": cv.std(),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "_prob": prob,
        })
        print(f"{name:20s} acc={results[-1]['accuracy']:.4f} roc_auc={results[-1]['roc_auc']:.4f} {params}")
    return results


def plot_results(results: list[dict], y_test, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, r in zip(axes.flat, results):
        heatmap(ax, r["confusion_matrix"], LABELS, LABELS)
        ax.set(title=f"{r['model']} (acc {r['accuracy']:.3f})", xlabel="predicted", ylabel="true")
    save(fig, out / "confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_prob"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC {r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves")
    ax.legend(loc="lower right")
    save(fig, out / "roc_curves.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "classification")
    args = parser.parse_args(argv)

    X, y = load_breast_cancer_df()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    args.out.mkdir(parents=True, exist_ok=True)
    plot_results(results, y_test, args.out)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    table = pd.DataFrame(clean).set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(4)
    table.to_csv(args.out / "metrics.csv")
    (args.out / "metrics.json").write_text(json.dumps(
        {"train_size": len(X_train), "test_size": len(X_test), "models": clean}, indent=2, default=float),
        encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
