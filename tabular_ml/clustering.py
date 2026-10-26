"""Unsupervised clustering of breast cancer samples: K-Means, agglomerative (Ward) and DBSCAN."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tabular_ml.data import RESULTS_DIR, load_breast_cancer_df
from tabular_ml.plotting import plt, save

SEED = 42


def preprocess(X: pd.DataFrame) -> np.ndarray:
    X = X.fillna(X.median())
    X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1)
    return StandardScaler().fit_transform(X)


def score(X, labels, true_labels) -> dict:
    n_clusters = len(set(labels) - {-1})
    result = {
        "clusters": n_clusters,
        "noise_points": int((labels == -1).sum()),
        "ARI": adjusted_rand_score(true_labels, labels),
        "NMI": normalized_mutual_info_score(true_labels, labels),
    }
    if n_clusters > 1:
        result["silhouette"] = silhouette_score(X, labels)
    return result


def choose_k(X, out: Path, k_max: int = 10) -> None:
    ks = range(1, k_max + 1)
    inertia = [KMeans(k, random_state=SEED, n_init=10).fit(X).inertia_ for k in ks]
    silhouettes = [silhouette_score(X, KMeans(k, random_state=SEED, n_init=10).fit_predict(X)) for k in ks[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, inertia, marker="o")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="inertia")
    axes[1].plot(ks[1:], silhouettes, marker="o")
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="silhouette")
    save(fig, out / "kmeans_k_selection.png")


def plot_dendrogram(X, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage(X, method="ward"), ax=ax, truncate_mode="level", p=6, no_labels=True)
    ax.set(title="Ward linkage dendrogram", ylabel="distance")
    save(fig, out / "dendrogram.png")


def plot_k_distance(X, out: Path, min_samples: int) -> None:
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(distances[:, -1]))
    ax.set(title=f"{min_samples}-distance graph (DBSCAN eps)", xlabel="points (sorted)", ylabel="distance")
    save(fig, out / "dbscan_k_distance.png")


def plot_clusters(X, labelings: dict, out: Path) -> None:
    pts = PCA(n_components=2, random_state=SEED).fit_transform(X)
    fig, axes = plt.subplots(1, len(labelings), figsize=(4.5 * len(labelings), 4.2))
    for ax, (name, labels) in zip(axes, labelings.items()):
        ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis", s=10)
        ax.set(title=name, xlabel="PC1", ylabel="PC2")
    save(fig, out / "clusters_pca.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "clustering")
    parser.add_argument("--eps", type=float, default=3.0)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    features, y = load_breast_cancer_df()
    X = preprocess(features)

    choose_k(X, args.out)
    plot_dendrogram(X, args.out)
    plot_k_distance(X, args.out, args.min_samples)

    labelings = {
        "K-Means (k=2)": KMeans(2, random_state=SEED, n_init=10).fit_predict(X),
        "Agglomerative (Ward, k=2)": AgglomerativeClustering(2, linkage="ward").fit_predict(X),
        f"DBSCAN (eps={args.eps})": DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(X),
    }
    metrics = {name: score(X, labels, y) for name, labels in labelings.items()}
    labelings["Ground truth"] = y.to_numpy()
    plot_clusters(X, labelings, args.out)

    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(pd.DataFrame(metrics).T.round(4))


if __name__ == "__main__":
    main()
