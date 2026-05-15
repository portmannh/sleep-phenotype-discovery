import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import umap.umap_ as umap

RANDOM_STATE = 42

clusterings = {
    "K-means": labels_kmeans,
    "Agglomerative": labels_agglo
}


def clustering_pipeline(feature_df, n_clusters, random_state):

    """
    Reduce dimensionality with UMAP, then cluster with different methods.

    Input:
    - feature_df: dataframe of features. Last column is the dataset of origin label
    - n_clusters: number of clusters
    - random_state: seed

    Output:
    - X_umap: features in UMAP space
    - clusterings: dict with clustering results
    """

    # remove labels
    X = feature_df.iloc[:,:-1]
    y = feature_df.iloc[:,-1]

    # standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP to reduce dimensionality to 2 before clustering
    umap_2d = umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.1, random_state=RANDOM_STATE)
    X_umap = umap_2d.fit_transform(X_scaled)


    # Agglomerative clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_agglo = agglo.fit_predict(X_umap)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state = 42, n_init = 5)
    labels_kmeans = kmeans.fit_predict(X_umap)

    clusterings = {
    "K-means": labels_kmeans,
    "Agglomerative": labels_agglo
    }

    return X_umap, clusterings


def silhouette_score(X_umap, clusterings):

    """
    Compute silhouette score of clustering results.

    Input:
    - X_umap: features in UMAP space
    - clusterings: dict with clustering results

    Output:
    - results_df: dataframe with silhouette score of each method
    """
    
    results = []

    for name, labels in clusterings.items():
        labels = np.array(labels)

        # For DBSCAN, exclude noise points for silhouette if needed
        if np.any(labels == -1):
            mask = labels != -1
            X_eval = X_umap[mask]
            labels_eval = labels[mask]
        else:
            X_eval = X_umap
            labels_eval = labels

        # Silhouette is only defined if there are at least 2 clusters
        unique_clusters = set(labels_eval)
        if len(unique_clusters) >= 2 and len(labels_eval) > len(unique_clusters):
            sil = silhouette_score(X_eval, labels_eval)
        else:
            sil = np.nan

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        results.append({
            "Method": name,
            "Clusters found": n_clusters_found,
            "Noise points": n_noise,
            "Silhouette score": sil
        })

    results_df = pd.DataFrame(results).sort_values(by="Silhouette score", ascending=False)

    return results_df

def plot_clusters(X_umap, clusterings, y_visualization=None, label_map=None, marker_map=None):

    """
    Plot clusters in UMAP space

    Input:
    - X_umap: features in UMAP space
    - clusterings: dict with clustering results
    - y_visualization: labels to overlay on the plot (e.g. dataset, sleep efficiency...)
    - label_map: dict with labels of y_visualization unique values
    - marker_map: dict with markers for y_visualization unique values

    """

    n_methods = len(clusterings.items())
        
    # 2 columns, enough rows to fit all methods
    ncols = 2
    nrows = math.ceil(n_methods / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows)
    )

    for ax, (name, labels) in zip(axes.ravel(), clusterings.items()):
        labels = np.array(labels)

        if y_visualization is not None:
            # Plot each true label with a different marker
            for y_val in np.unique(y_visualization):

                marker = marker_map.get(y_val, "o")

                # Regular clustered points
                mask_regular = (labels != -1) & (y_visualization == y_val)

                ax.scatter(
                    X_umap[mask_regular, 0],
                    X_umap[mask_regular, 1],
                    c=labels[mask_regular],
                    marker=marker,
                    alpha=0.7,
                    label=f"{label_map[y_val]}"
                )

                # Noise points
                mask_noise = (labels == -1) & (y_visualization == y_val)

                if np.any(mask_noise):
                    ax.scatter(
                        X_umap[mask_noise, 0],
                        X_umap[mask_noise, 1],
                        marker="x",
                        s=60,
                        color="black"
                    )

            # Custom legend with black symbols
            legend_elements = [
                Line2D(
                    [0], [0],
                    marker=marker_map.get(y_val, "o"),
                    color="black",
                    linestyle="None",
                    markersize=8,
                    label=f"{label_map[y_val]}"
                )
                for y_val in np.unique(y_visualization)
            ]

            ax.legend(handles=legend_elements)

        else:
            labels = np.array(labels)

            # Plot non-noise points first
            mask_regular = labels != -1
            sc = ax.scatter(
                X_umap[mask_regular, 0],
                X_umap[mask_regular, 1],
                c=labels[mask_regular],
                alpha=0.7
            )

            # Highlight noise / outliers if present
            mask_noise = labels == -1
            if np.any(mask_noise):
                ax.scatter(
                    X_umap[mask_noise, 0],
                    X_umap[mask_noise, 1],
                    marker="x",
                    s=60,
                    label="Noise / outlier"
                )
            ax.legend()

        ax.set_title(name)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True)

    plt.tight_layout()

    plt.savefig("clusters.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1)
    plt.savefig("clusters.svg",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1)

    plt.show()






