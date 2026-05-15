import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import random

random.seed(42)


def run_dbscan_clustering(feature_matrix, title, eps=1.5, minPts=5, index=None):
    """
    Run DBSCAN on a feature matrix (DataFrame or 2D array, e.g. UMAP coordinates).

    Parameters
    ----------
    feature_matrix : pd.DataFrame or array-like of shape (n_samples, n_features)
    title : str
        Printed header for cluster-count summary.
    eps : float
        DBSCAN neighbourhood radius.
    minPts : int
        DBSCAN ``min_samples``.
    index : array-like, optional
        Row index when ``feature_matrix`` is a NumPy array (e.g. ``hypno_feature_df.index``).

    Returns
    -------
    clustered : pd.DataFrame
        Input features (or UMAP1/UMAP2) plus ``dbscan_label``.
    labels : np.ndarray
        Cluster labels (-1 = noise).
    X_fit : np.ndarray
        Matrix passed to DBSCAN (same as input if already numeric 2D).
    """
    if isinstance(feature_matrix, pd.DataFrame):
        X_fit = feature_matrix.to_numpy(dtype=float)
        row_index = feature_matrix.index
    else:
        X_fit = np.asarray(feature_matrix, dtype=float)
        if X_fit.ndim != 2:
            raise ValueError("feature_matrix must be 2D (n_samples, n_features).")
        row_index = index
        if row_index is None:
            row_index = np.arange(len(X_fit))

    labels = DBSCAN(eps=eps, min_samples=minPts, metric="euclidean").fit_predict(X_fit)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"{title}")
    print(f"  Clusters (excluding noise): {n_clusters}")
    print(f"  Noise points: {n_noise}")
    print(pd.Series(labels, name="dbscan_label").value_counts().sort_index())

    if isinstance(feature_matrix, pd.DataFrame):
        clustered = feature_matrix.copy()
    else:
        cols = ["UMAP1", "UMAP2"] if X_fit.shape[1] == 2 else [f"dim_{i}" for i in range(X_fit.shape[1])]
        clustered = pd.DataFrame(X_fit, index=row_index, columns=cols)

    clustered["dbscan_label"] = labels

    return clustered, labels, X_fit


def plot_cluster_embedding(coords, labels, title, xlabel="Dim 1", ylabel="Dim 2"):
    """Scatter plot of 2D embedding coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        s=45,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="cluster")
    plt.tight_layout()
    plt.show()
