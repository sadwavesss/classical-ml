import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from tabular_ml.classification import QuantileClipper, build_models, train_and_evaluate
from tabular_ml.clustering import preprocess, score
from tabular_ml.data import load_breast_cancer_df
from tabular_ml.eda import iqr_filter
from tabular_ml.regression import prepare
from tabular_ml.tabular_mlp import MLP, encode, predict_proba, train


def test_quantile_clipper_uses_training_quantiles():
    train_x = np.arange(101, dtype=float).reshape(-1, 1)
    clipper = QuantileClipper(0.1, 0.9).fit(train_x)
    assert clipper.transform([[-50.0], [50.0], [500.0]]).ravel().tolist() == [10.0, 50.0, 90.0]


def test_iqr_filter_drops_extreme_rows():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 1000], "b": [1, 1, 1, 1, 1, 1]})
    assert iqr_filter(df, ["a", "b"])["a"].max() == 5


def test_regression_prepare_removes_zero_target_and_encodes_genre():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "popularity": np.r_[np.zeros(5), rng.integers(30, 90, 95)],
        "energy": rng.random(100),
        "tempo": rng.normal(120, 10, 100),
        "genre": rng.choice(["pop", "rock", "hip hop"], 100),
    })
    X, y = prepare(df, n_lowest=10)
    assert (y > 0).all() and len(y) <= 85
    assert {"genre_pop", "genre_rock"} <= set(X.columns)


@pytest.fixture(scope="module")
def cancer_split():
    X, y = load_breast_cancer_df()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def test_logistic_regression_is_accurate(cancer_split):
    models = {"Logistic Regression": build_models()["Logistic Regression"]}
    [result] = train_and_evaluate(*cancer_split, models=models)
    assert result["accuracy"] > 0.93 and result["roc_auc"] > 0.98


def test_cluster_score_handles_single_cluster():
    X = preprocess(load_breast_cancer_df()[0])
    y = load_breast_cancer_df()[1]
    result = score(X, np.zeros(len(X), dtype=int), y)
    assert result["clusters"] == 1 and "silhouette" not in result
    assert result["ARI"] == 0


def test_mlp_learns_separable_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=600), "x2": rng.normal(size=600),
                      "cat": rng.choice(["a", "b"], 600)})
    y = pd.Series(np.where(X["x1"] + X["x2"] > 0, ">50K", "<=50K"))
    X_train, X_test, y_train, y_test, classes = encode(X, y)
    assert classes == ["<=50K", ">50K"] and X_train.shape[1] == 4

    model, best, best_epoch, history = train(X_train, X_test, y_train, y_test, epochs=15, hidden_dim=16, lr=1e-2)
    assert isinstance(best, MLP) and 1 <= best_epoch <= 15
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert min(history["val_loss"]) == history["val_loss"][best_epoch - 1]
    prob = predict_proba(best, X_test)
    assert prob.shape == (len(y_test),) and ((prob >= 0.5) == y_test).mean() > 0.9
