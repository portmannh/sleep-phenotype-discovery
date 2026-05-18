import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
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


def _dbscan_cluster_colors(labels, palette="Set2", noise_color="#b0b0b0"):
    """Map DBSCAN labels to Set2 colors (same order as violin plots: Cluster 0, 1, …)."""
    labels = np.asarray(labels)
    clusters = sorted(int(x) for x in np.unique(labels) if x >= 0)
    palette_colors = sns.color_palette(palette, n_colors=max(len(clusters), 1))
    color_by_label = {-1: noise_color}
    for i, cluster_id in enumerate(clusters):
        color_by_label[cluster_id] = palette_colors[i]
    point_colors = [color_by_label[int(lbl)] for lbl in labels]
    return point_colors, color_by_label


def plot_cluster_embedding(coords, labels, title, xlabel="UMAP 1", ylabel="UMAP 2"):
    """Scatter plot of 2D coordinates coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        np.asarray(coords)[:, 0],
        np.asarray(coords)[:, 1],
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


def plot_dbscan_panels(
    results,
    suptitle="DBSCAN on UMAP embeddings",
    overlay_match=None,
    overlay_marker="x",
    overlay_label=None,
    overlay_cohort=None,
    cluster_palette="Set2",
):
    """Plot one row of UMAP scatter panels coloured by DBSCAN labels.

    Parameters
    ----------
    results : list of (coords, labels, title) or (coords, labels, title, point_labels)
        Optional 4th item: per-point labels for the overlay (e.g. ``dataset`` or ``normal_sme``).
    overlay_match : optional
        Points where ``point_labels == overlay_match`` get ``overlay_marker`` on top.
        Examples: ``'sc'`` (sleep-cassette), ``False`` (low SME). Pass ``None`` to disable.
    overlay_marker : str
        Matplotlib marker for the overlay (default ``'x'``).
    overlay_label : str, optional
        Legend text; inferred for bool ``overlay_match`` if omitted.
    overlay_cohort : optional
        Deprecated alias for ``overlay_match``.
    cluster_palette : str
        Seaborn palette for DBSCAN clusters (default ``Set2``, same as violin plots).
    """
    if overlay_cohort is not None:
        overlay_match = overlay_cohort

    def _resolve_overlay_match(point_labels, match):
        """Infer overlay target when 4th column is passed but overlay_match is omitted."""
        if match is not None or point_labels is None:
            return match
        series = pd.Series(point_labels)
        if series.dtype == bool:
            return False
        values = set(series.astype(str).str.strip().unique())
        if "sc" in values:
            return "sc"
        return match

    def _overlay_legend_label(match):
        if match is False:
            return "low SME (<70%)"
        if match is True:
            return "normal SME (≥70%)"
        if match == "sc":
            return "sleep-cassette"
        if match is not None:
            return str(match)
        return "overlay"
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    axes = np.atleast_1d(axes).ravel().tolist()

    for ax, item in zip(axes, results):
        if len(item) == 4:
            coords, labels, title, point_labels = item
        else:
            coords, labels, title = item
            point_labels = None

        if isinstance(coords, pd.DataFrame):
            if {"UMAP1", "UMAP2"}.issubset(coords.columns):
                plot_xy = coords[["UMAP1", "UMAP2"]].to_numpy(dtype=float)
            else:
                plot_xy = coords.iloc[:, :2].to_numpy(dtype=float)
        else:
            plot_xy = np.asarray(coords, dtype=float)

        labels = np.asarray(labels)
        if plot_xy.ndim != 2 or plot_xy.shape[1] < 2:
            raise ValueError(
                f"coords must be (n_samples, 2); got shape {plot_xy.shape}. "
                "Pass UMAP coordinates (e.g. hypno_umap or clustered[['UMAP1', 'UMAP2']])."
            )

        point_colors, color_by_label = _dbscan_cluster_colors(
            labels, palette=cluster_palette
        )
        ax.scatter(
            plot_xy[:, 0],
            plot_xy[:, 1],
            c=point_colors,
            s=45,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.3,
            zorder=2,
        )

        legend_handles = [
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                markerfacecolor=color_by_label[label],
                markeredgecolor="k",
                markersize=8,
                label=f"Cluster {label}" if label >= 0 else "Noise",
            )
            for label in sorted(color_by_label)
        ]

        panel_overlay = _resolve_overlay_match(point_labels, overlay_match)

        if point_labels is not None and panel_overlay is not None:
            point_labels = pd.Series(point_labels)
            if len(point_labels) != len(plot_xy):
                raise ValueError(
                    f"point_labels length ({len(point_labels)}) must match coords ({len(plot_xy)}). "
                    "Slice per panel, e.g. sme_labels.loc[hypno_clustered.index, 'normal_sme']."
                )
            if point_labels.dtype == bool or panel_overlay in (True, False):
                mask = point_labels.to_numpy() == panel_overlay
            else:
                mask = (
                    point_labels.astype(str).str.strip().to_numpy()
                    == str(panel_overlay).strip()
                )
            if np.any(mask):
                ax.plot(
                    plot_xy[mask, 0],
                    plot_xy[mask, 1],
                    linestyle="",
                    marker=overlay_marker,
                    color="k",
                    markersize=9,
                    markeredgewidth=1.8,
                    zorder=10,
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="",
                        marker=overlay_marker,
                        markerfacecolor="none",
                        markeredgecolor="k",
                        markersize=8,
                        label=overlay_label or _overlay_legend_label(panel_overlay),
                    )
                )

        ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.3)
        ax.autoscale()

    fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.show()
