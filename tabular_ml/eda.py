"""Exploratory analysis of the Spotify top hits dataset (2000-2019)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_ml.data import RESULTS_DIR, load_spotify
from tabular_ml.plotting import heatmap, plt, save


def iqr_filter(df: pd.DataFrame, columns) -> pd.DataFrame:
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    return df


def summarize(df: pd.DataFrame) -> dict:
    by_year = df.groupby("year")["popularity"].mean()
    return {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "most_common_genre": df["genre"].value_counts().idxmax(),
        "top_artists": df["artist"].value_counts().head(5).to_dict(),
        "popularity_mean": round(df["popularity"].mean(), 2),
        "popularity_median": float(df["popularity"].median()),
        "mean_duration_sec": round(df["duration_ms"].mean() / 1000, 1),
        "long_tracks_since_2000": int(((df["year"] >= 2000) & (df["duration_ms"] >= 300_000)).sum()),
        "popularity_by_year_last10": by_year.tail(10).round(1).to_dict(),
        "most_danceable_genres": df.groupby("genre")["danceability"].mean().nlargest(5).round(3).to_dict(),
        "most_acoustic_genres": df.groupby("genre")["acousticness"].mean().nlargest(5).round(3).to_dict(),
        "energy_loudness_corr": round(df["energy"].corr(df["loudness"]), 3),
        "tempo_range": [round(df["tempo"].min(), 1), round(df["tempo"].max(), 1)],
    }


def plot_distributions(df: pd.DataFrame, out) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(df["popularity"], bins=30, color="tab:blue")
    axes[0, 0].set_title("Popularity")
    axes[0, 1].hist(df["danceability"], bins=30, color="tab:orange")
    axes[0, 1].set_title("Danceability")
    axes[1, 0].boxplot(df["duration_ms"] / 1000, vert=False)
    axes[1, 0].set_title("Duration, s")
    df["genre"].value_counts().head(10).plot.barh(ax=axes[1, 1], color="tab:green")
    axes[1, 1].set_title("Top-10 genres")
    axes[1, 1].invert_yaxis()
    save(fig, out / "distributions.png")


def plot_correlations(df: pd.DataFrame, out) -> None:
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = heatmap(ax, corr.values, corr.columns, corr.columns, fmt="{:.2f}", cmap="coolwarm")
    im.set_clim(-1, 1)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation matrix")
    save(fig, out / "correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(df["energy"], df["loudness"], s=8, alpha=0.5)
    axes[0].set(xlabel="energy", ylabel="loudness, dB", title="Energy vs loudness")
    axes[1].scatter(df["tempo"], df["danceability"], s=8, alpha=0.5, color="tab:orange")
    axes[1].set(xlabel="tempo, BPM", ylabel="danceability", title="Tempo vs danceability")
    save(fig, out / "scatter.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "eda")
    args = parser.parse_args(argv)

    raw = load_spotify(args.data)
    df = raw.drop_duplicates()
    numeric = df.select_dtypes(include=np.number).columns
    clean = iqr_filter(df, numeric)

    summary = summarize(df)
    summary["duplicates_removed"] = len(raw) - len(df)
    summary["rows_after_iqr_filter"] = len(clean)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    plot_distributions(df, out)
    plot_correlations(df, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
