"""Predicting track popularity from audio features: linear, polynomial, Ridge and Lasso regression."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from tabular_ml.data import RESULTS_DIR, load_spotify
from tabular_ml.plotting import plt, save

SEED = 42
TARGET = "popularity"


def prepare(df: pd.DataFrame, z_threshold: float = 4.0, n_lowest: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    numeric = df.select_dtypes(include=np.number).columns.drop(TARGET)
    genres = pd.get_dummies(df["genre"], prefix="genre", drop_first=True, dtype=float)
    X = pd.concat([df[numeric], genres], axis=1)
    y = df[TARGET]

    # Zero and near-zero popularity values sit far below the rest of the target and are excluded.
    keep = y > 0
    X, y = X[keep], y[keep]
    drop = y.nsmallest(min(n_lowest, len(y))).index
    X, y = X.drop(index=drop), y.drop(index=drop)

    X = X.fillna(X.median())
    numeric_part = X[numeric]
    outliers = (np.abs(stats.zscore(numeric_part)) > z_threshold).any(axis=1)
    return X[~outliers].reset_index(drop=True), y[~outliers].reset_index(drop=True)


def build_models(numeric: list[str]) -> dict:
    # Polynomial terms only for audio features; genre dummies pass through. Alpha is chosen by CV on train.
    poly = ColumnTransformer([("poly", make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False)),
                               numeric)], remainder="passthrough")
    return {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial (deg=2) + RidgeCV": make_pipeline(poly, StandardScaler(),
                                                      RidgeCV(alphas=np.logspace(-1, 4, 21))),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=10_000, random_state=SEED)),
    }


def evaluate(models: dict, X_train, X_test, y_train, y_test):
    rows, predictions = [], {}
    for name, model in models.items():
        pred = model.fit(X_train, y_train).predict(X_test)
        predictions[name] = pred
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })
    baseline = np.full_like(y_test, y_train.mean(), dtype=float)
    rows.append({"model": "Mean baseline", "MAE": mean_absolute_error(y_test, baseline),
                 "RMSE": mean_squared_error(y_test, baseline) ** 0.5, "R2": r2_score(y_test, baseline)})
    return pd.DataFrame(rows).set_index("model").round(4), predictions


def plot_predictions(y_test, predictions: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (name, pred) in zip(axes.flat, predictions.items()):
        ax.scatter(y_test, pred, s=10, alpha=0.6)
        ax.plot([0, 100], [0, 100], "k--", linewidth=1)
        ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="true popularity", ylabel="predicted", title=name)
    save(fig, out / "predictions.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "regression")
    args = parser.parse_args(argv)

    X, y = prepare(load_spotify(args.data).drop_duplicates())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    numeric = [c for c in X.columns if not c.startswith("genre_")]
    metrics, predictions = evaluate(build_models(numeric), X_train, X_test, y_train, y_test)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "metrics.csv")
    (args.out / "metrics.json").write_text(json.dumps(
        {"train_size": len(X_train), "test_size": len(X_test), "features": X.shape[1],
         "metrics": metrics.to_dict(orient="index")}, indent=2), encoding="utf-8")
    plot_predictions(y_test, predictions, args.out)
    print(metrics.sort_values("R2", ascending=False))


if __name__ == "__main__":
    main()
