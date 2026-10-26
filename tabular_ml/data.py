from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SPOTIFY_URL = "https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019"
ADULT_NUMERIC = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]


def load_spotify(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_DIR / "spotify_top_hits.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download the dataset from {SPOTIFY_URL}")
    return pd.read_csv(path)


def load_breast_cancer_df() -> tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn (0 = malignant, 1 = benign)."""
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target


def load_adult(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """UCI Adult census income. Uses data/adult.csv if present, otherwise downloads it from OpenML."""
    path = Path(path) if path else DATA_DIR / "adult.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = fetch_openml("adult", version=2, as_frame=True).frame
        df.columns = [c.replace("-", ".") for c in df.columns]
        df = df.rename(columns={"class": "income"})

    y = df["income"].astype(str).str.strip().str.rstrip(".")
    X = df.drop(columns=["income"]).copy()
    for col in X.columns:
        if col in ADULT_NUMERIC:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown"})
    return X, y
