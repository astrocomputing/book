**Chapter 23: Unsupervised Learning: Clustering and Dimensionality Reduction**

Shifting focus from supervised learning (where labeled data guides the prediction of outputs), this chapter delves into **unsupervised learning**, a branch of machine learning that aims to uncover hidden patterns, structures, and relationships within **unlabeled data**. Without predefined target labels, unsupervised algorithms explore the intrinsic properties of the data itself. We will concentrate on two primary categories of unsupervised learning highly relevant to astrophysical exploration: **clustering**, which involves grouping similar data points together based on their features, and **dimensionality reduction**, which aims to simplify complex, high-dimensional datasets by projecting them into a lower-dimensional space while retaining essential information. We will explore common clustering algorithms, including the centroid-based **K-Means**, the density-based **DBSCAN**, and **Hierarchical Clustering**, discussing their principles, parameters, and implementation using `scikit-learn.cluster`. Methods for evaluating the quality of clustering results, such as the Silhouette Score, will be introduced. Subsequently, we will cover dimensionality reduction techniques, focusing on the widely used linear method **Principal Component Analysis (PCA)** for finding axes of maximum variance (`sklearn.decomposition.PCA`), and briefly introducing powerful non-linear **manifold learning** techniques like **t-SNE** and **UMAP** (`sklearn.manifold.TSNE`, `umap-learn`) which are particularly effective for visualizing the underlying structure of high-dimensional data in 2D or 3D plots.

**23.1 Finding Structure in Unlabeled Data**

Unsupervised learning addresses a fundamentally different goal compared to supervised learning. Instead of learning a mapping from input features `X` to known output labels `y`, unsupervised algorithms work solely with the input features `X` of a dataset where labels are either unavailable, expensive to obtain, or intentionally ignored. The primary objective is **exploratory data analysis** – to discover inherent structures, groupings, patterns, or simpler representations within the data itself without prior guidance about what those structures might be. This makes unsupervised learning a powerful tool for scientific discovery, anomaly detection, and data preprocessing in astrophysics.

Why is finding structure in unlabeled data important in astronomy? Firstly, astronomical surveys often generate vast catalogs of objects (stars, galaxies) with measured properties (features) but lacking definitive classifications (labels) for every object. Applying unsupervised **clustering** algorithms to this data, perhaps based on photometric colors, kinematic properties (from Gaia), or morphological parameters, can automatically identify groups of objects that share similar characteristics. These data-driven groups might correspond to known astrophysical populations (like star clusters, stellar streams, specific types of galaxies) or potentially reveal previously unrecognized sub-classes or associations worthy of further investigation. Clustering provides a way to organize and segment large datasets based on intrinsic similarities.

Secondly, modern astrophysical datasets are often **high-dimensional**. A single galaxy spectrum might contain thousands of flux measurements at different wavelengths; a simulation snapshot might track dozens of properties for billions of particles; a catalog might combine measurements from multiple surveys across different wavelengths. Visualizing and interpreting relationships within such high-dimensional spaces is extremely challenging for humans. **Dimensionality reduction** techniques provide a way to project this high-dimensional data onto a lower-dimensional space (typically 2D or 3D) while attempting to preserve the most important structures or relationships present in the original data. This allows for visualization of complex datasets, potentially revealing clusters, manifolds, or outliers that were hidden in the high-dimensional view. It can also be used as a preprocessing step (**feature extraction**) to create a smaller set of more informative input features for subsequent supervised learning tasks, potentially improving performance and reducing computational cost.

A third application of unsupervised learning is **anomaly detection** or **outlier detection**. By identifying data points that do not conform well to the general structure or clusters found in the bulk of the data, unsupervised methods can flag potentially interesting outliers. These outliers might represent rare or novel astrophysical phenomena, unusual evolutionary states, measurement errors, or instrumental artifacts. Algorithms like DBSCAN (Sec 23.2) naturally identify noise points, and other specialized outlier detection methods exist.

Unlike supervised learning where performance is easily measured by comparing predictions to known labels, evaluating the success of unsupervised learning can be more subjective and context-dependent. For clustering, evaluation might involve internal metrics that measure cluster cohesion and separation (like Silhouette Score, Sec 23.3), external validation by comparing clusters to known (but unused) labels if available, or ultimately, assessing whether the discovered clusters correspond to scientifically meaningful groupings. For dimensionality reduction, evaluation often focuses on how well the low-dimensional representation preserves the structure (e.g., local neighborhoods, global distances) of the original data, often assessed visually or through specific preservation metrics.

The algorithms used in unsupervised learning typically rely on measuring **similarity or distance** between data points in the feature space. Therefore, preprocessing steps like **feature scaling** (Standardization or Normalization, Sec 20.2) are often even *more critical* for unsupervised algorithms (especially distance-based ones like K-Means, DBSCAN, PCA) than for some supervised ones, ensuring that features with larger numerical ranges do not unduly dominate the similarity calculations.

This chapter will explore specific algorithms for clustering and dimensionality reduction. We begin with clustering methods – K-Means, DBSCAN, and Hierarchical Clustering – which aim to partition the data into meaningful groups. We then move to dimensionality reduction, covering the foundational linear technique PCA and the powerful non-linear visualization techniques t-SNE and UMAP. Understanding these unsupervised tools provides essential capabilities for exploring complex, unlabeled astrophysical datasets and discovering the structures hidden within.

**23.2 Clustering Algorithms: K-Means, DBSCAN, Hierarchical Clustering**

**Clustering** is the task of grouping a set of objects (data points) such that objects in the same group (called a **cluster**) are more similar (according to some distance or similarity measure in feature space) to each other than to those in other clusters. It's a core task in unsupervised learning, aiming to discover natural groupings within unlabeled data. `scikit-learn.cluster` provides implementations of many common clustering algorithms. We will discuss three popular approaches: K-Means, DBSCAN, and Hierarchical Clustering.

**1. K-Means Clustering:** K-Means is one of the simplest and most widely used clustering algorithms. It aims to partition `n` data points into `k` predefined clusters, where `k` is a hyperparameter specified by the user. The algorithm works iteratively:
*   **Initialization:** Randomly select `k` initial cluster centroids (representative points) in the feature space.
*   **Assignment Step:** Assign each data point to the cluster whose centroid is nearest (typically using Euclidean distance).
*   **Update Step:** Recalculate the position of each centroid as the mean (centroid) of all data points assigned to that cluster.
*   **Iteration:** Repeat the Assignment and Update steps until the centroids no longer move significantly or a maximum number of iterations is reached.
`sklearn.cluster.KMeans(n_clusters=k, ...)` implements this. Key parameters include `n_clusters` (the number of clusters `k` to find) and `n_init` (number of times to run with different random initial centroids, returning the best result to mitigate sensitivity to initialization).

K-Means is computationally efficient (especially its variants like MiniBatchKMeans) and easy to implement. However, it has limitations:
*   It requires the user to **specify the number of clusters `k` beforehand**, which might not be known *a priori*. Choosing `k` often involves running the algorithm for different `k` values and evaluating cluster quality using metrics like the Silhouette Score (Sec 23.3) or the "elbow method" (plotting within-cluster variance vs. `k`).
*   It assumes clusters are **spherical, equally sized, and have similar densities**, as it uses the mean as the centroid and Euclidean distance. It struggles with elongated clusters, clusters of different sizes, non-convex shapes, or varying densities.
*   It is sensitive to the **initial placement of centroids** and can converge to local optima; running multiple initializations (`n_init='auto'` or a specific number > 1 in scikit-learn) is recommended.
*   Feature **scaling** (e.g., `StandardScaler`) is generally required before applying K-Means, as it relies on Euclidean distances.

**2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** DBSCAN offers a fundamentally different approach based on the **density** of data points. It groups together points that are closely packed, marking points that lie alone in low-density regions as outliers (noise). It does *not* require specifying the number of clusters beforehand. DBSCAN works with two key hyperparameters:
*   `eps` (epsilon): A distance threshold. Two points are considered neighbors if the distance between them is less than or equal to `eps`.
*   `min_samples`: The minimum number of points required within a point's `eps`-neighborhood (including the point itself) for it to be considered a **core point**.
The algorithm proceeds as follows:
*   Randomly select an unvisited point `P`.
*   If `P` is a core point (has ≥ `min_samples` neighbors within `eps`), start a new cluster. Add `P` and all its density-reachable neighbors (neighbors that are also core points, and their neighbors, etc.) to this cluster.
*   If `P` is not a core point, mark it (temporarily) as noise. It might later be assigned to a cluster if it's found to be a neighbor of a core point (a "border point").
*   Repeat until all points are visited.
`sklearn.cluster.DBSCAN(eps=..., min_samples=...)` implements this.

DBSCAN's advantages include:
*   It does **not require specifying `k`**. The number of clusters is determined automatically based on density.
*   It can find **arbitrarily shaped clusters** (not just spherical ones).
*   It is robust to **outliers**, explicitly identifying them as noise (typically labeled -1).
Disadvantages include:
*   It can be sensitive to the choice of `eps` and `min_samples`, which might require tuning and can be difficult to choose for datasets with varying densities.
*   It struggles with clusters of **significantly different densities**, as a single `eps`/`min_samples` setting might merge dense clusters or fail to identify sparse ones.
*   Feature **scaling** is crucial, as it relies on distance calculations defined by `eps`.

**3. Hierarchical Clustering:** This approach builds a hierarchy of clusters, represented visually as a **dendrogram**. There are two main types:
*   **Agglomerative (Bottom-up):** Starts with each data point as its own cluster. In each step, it merges the two closest clusters based on a chosen **linkage criterion** (how cluster distance is defined, e.g., Ward, average, complete, single linkage) until all points belong to a single cluster.
*   **Divisive (Top-down):** Starts with all data points in one cluster and recursively splits clusters into smaller ones until each point is its own cluster. Agglomerative clustering is more common.
`sklearn.cluster.AgglomerativeClustering(n_clusters=k, linkage='ward', ...)` implements the agglomerative approach. You can specify the desired number of clusters `n_clusters` (which corresponds to cutting the dendrogram at a certain level), or specify a distance threshold `distance_threshold` to cut the dendrogram. Different `linkage` criteria ('ward', 'average', 'complete', 'single') determine which clusters are merged based on minimizing variance (Ward), average distance, maximum distance, or minimum distance between points in the clusters, respectively, leading to potentially different cluster shapes and structures.

Advantages of hierarchical clustering include:
*   It does **not require specifying `k` beforehand** if using a distance threshold or analyzing the dendrogram. The dendrogram itself provides a visualization of the nested cluster structure at different scales.
*   It can reveal hierarchical relationships within the data.
Disadvantages include:
*   It is computationally expensive, typically scaling as O(n²) or O(n³) for `n` data points, making it unsuitable for very large datasets (though variants exist).
*   The choice of linkage criterion can significantly impact the results (e.g., 'single' linkage can find non-convex shapes but is sensitive to noise; 'complete' or 'ward' often find more compact, spherical clusters).
*   Feature **scaling** is usually recommended as distance calculations are involved.

```python
# --- Code Example: Applying K-Means, DBSCAN, Agglomerative Clustering ---
# Note: Requires scikit-learn installation.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs # For example datasets

print("Applying different clustering algorithms:")

# --- Generate Sample Data ---
# Data 1: Blobs (good for K-Means, maybe Ward linkage)
X_blobs, y_blobs_true = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=0)
X_blobs_scaled = StandardScaler().fit_transform(X_blobs)

# Data 2: Moons (non-convex, good for DBSCAN)
X_moons, y_moons_true = make_moons(n_samples=300, noise=0.1, random_state=0)
X_moons_scaled = StandardScaler().fit_transform(X_moons)

# --- Apply K-Means ---
print("\nApplying K-Means (on Blobs data)...")
kmeans = KMeans(n_clusters=4, n_init='auto', random_state=0)
kmeans_labels = kmeans.fit_predict(X_blobs_scaled)
print(f"  K-Means found clusters for blobs data.")

# --- Apply DBSCAN ---
print("\nApplying DBSCAN (on Moons data)...")
# Parameters eps and min_samples often require tuning
dbscan = DBSCAN(eps=0.3, min_samples=5) 
dbscan_labels = dbscan.fit_predict(X_moons_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = np.sum(dbscan_labels == -1)
print(f"  DBSCAN found {n_clusters_db} clusters and {n_noise} noise points for moons data.")

# --- Apply Agglomerative Clustering ---
print("\nApplying Agglomerative Clustering (on Blobs data)...")
# Specify number of clusters or distance threshold
agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
agglo_labels = agglo.fit_predict(X_blobs_scaled)
print(f"  Agglomerative Clustering found clusters for blobs data.")

# --- Visualize Results (Conceptual) ---
print("\nGenerating comparison plots (conceptual)...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot K-Means results on blobs
axes[0].scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=10)
axes[0].set_title(f"K-Means (k=4) on Blobs")

# Plot DBSCAN results on moons
axes[1].scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=dbscan_labels, cmap='viridis', s=10)
axes[1].set_title(f"DBSCAN (eps=0.3) on Moons (Noise=-1)")

# Plot Agglomerative results on blobs
axes[2].scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=agglo_labels, cmap='viridis', s=10)
axes[2].set_title(f"Agglomerative (k=4, Ward) on Blobs")

for ax in axes: ax.set_xlabel("Scaled Feature 1"); ax.set_ylabel("Scaled Feature 2")
fig.tight_layout()
# plt.show()
print("Plots generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code demonstrates applying three different clustering algorithms.
# 1. It generates two synthetic datasets: 'blobs' (well-separated, spherical clusters) 
#    and 'moons' (non-convex shapes). Data is scaled using StandardScaler.
# 2. K-Means: It applies KMeans with `n_clusters=4` to the scaled blobs data. 
#    `fit_predict` both trains the model and returns the cluster label for each point.
# 3. DBSCAN: It applies DBSCAN with chosen `eps` and `min_samples` to the scaled moons data. 
#    It calculates the number of clusters found (excluding noise points labeled -1). 
#    DBSCAN is expected to perform better than K-Means on the moons dataset.
# 4. Agglomerative Clustering: It applies AgglomerativeClustering with `n_clusters=4` 
#    and 'ward' linkage (which tends to find equal-variance clusters, suitable for blobs) 
#    to the scaled blobs data.
# 5. Visualization: It creates scatter plots showing the data points colored by the cluster 
#    labels assigned by each algorithm, visually demonstrating how K-Means and Agglomerative 
#    work well for blobs, while DBSCAN correctly separates the non-convex moons.
```

The choice of clustering algorithm should be guided by the expected cluster shapes, the presence of noise, whether the number of clusters is known, and the size of the dataset. K-Means is simple and efficient but assumes spherical clusters. DBSCAN handles arbitrary shapes and noise but needs careful parameter tuning and struggles with varying densities. Hierarchical clustering provides a useful dendrogram visualization but scales poorly. Experimenting with different algorithms and parameters, combined with appropriate evaluation metrics and visualization, is often necessary to find the most meaningful clusters in unlabeled astrophysical data.

**23.3 Evaluating Clustering Performance**

Unlike supervised learning, where model performance can be clearly evaluated by comparing predictions against known true labels, evaluating the quality of **unsupervised clustering** is inherently more challenging and often subjective. Since there are no ground truth labels to compare against, evaluation typically relies on **internal metrics** that assess the geometric properties of the clusters found (e.g., how compact and well-separated they are) or **external metrics** that compare the clustering results to some known, external classification (like existing scientific labels), if available (though using external labels blurs the line with supervised evaluation).

**Internal Validation Metrics:** These metrics evaluate the clustering structure based solely on the data points and the cluster assignments produced by the algorithm, without reference to any external information. A common goal is to find clusters that are internally cohesive (points within a cluster are close to each other) and externally well-separated (clusters are far apart from each other).

One of the most popular internal metrics is the **Silhouette Score**. For each data point `i`, its silhouette coefficient `s(i)` measures how similar it is to its own cluster compared to other clusters. It is calculated as:
s(i) = [ b(i) - a(i) ] / max{a(i), b(i)}
where:
*   `a(i)` is the average distance from point `i` to all *other* points within the *same* cluster (measuring cohesion).
*   `b(i)` is the average distance from point `i` to all points in the *nearest neighboring cluster* (the closest cluster to which `i` does not belong, measuring separation).
The silhouette coefficient `s(i)` ranges from -1 to +1:
*   `s(i)` close to +1 indicates the point is well-clustered, far from neighboring clusters.
*   `s(i)` close to 0 indicates the point is close to a decision boundary between clusters.
*   `s(i)` close to -1 indicates the point might have been misclassified and belongs better in the neighboring cluster.
The **Silhouette Score** for the entire clustering solution is the *average* `s(i)` over all data points. A score closer to +1 indicates a better-defined clustering structure (dense, well-separated clusters). `sklearn.metrics.silhouette_score(X, labels)` calculates the average score given the feature data `X` and the cluster `labels` assigned by the algorithm. It's often used to help select the optimal number of clusters `k` in algorithms like K-Means (by choosing the `k` that maximizes the Silhouette Score).

Another internal metric is the **Davies-Bouldin Index**. It measures the ratio of within-cluster scatter to between-cluster separation. Lower values indicate better clustering (clusters are compact and well-separated). `sklearn.metrics.davies_bouldin_score(X, labels)` calculates this index.

The **Calinski-Harabasz Index** (also known as the Variance Ratio Criterion) measures the ratio of between-cluster variance to within-cluster variance. Higher values indicate better clustering (larger separation between clusters relative to within-cluster dispersion). `sklearn.metrics.calinski_harabasz_score(X, labels)` calculates this.

Internal metrics provide a quantitative way to compare different clustering results (e.g., from different algorithms or different `k` values for K-Means) based purely on the data geometry. However, they make implicit assumptions about what constitutes a "good" cluster (often favoring compact, spherical clusters) and may not always align perfectly with scientifically meaningful groupings, especially for complex cluster shapes or varying densities where methods like DBSCAN might be more appropriate but might yield lower scores on metrics favoring compactness.

**External Validation Metrics:** If, for testing or comparison purposes, you have access to *true* class labels for your data (even though the clustering algorithm didn't use them), you can use external validation metrics to assess how well the discovered clusters correspond to these known classes. These metrics treat the clustering result as a prediction and compare it to the ground truth classification.

Common external metrics include:
*   **Adjusted Rand Index (ARI):** Measures the similarity between the true labels and the clustering labels, correcting for chance. Ranges from -1 (independent labelings) to +1 (perfect match). Values close to 1 indicate the clusters align well with the true classes. `sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)`.
*   **Normalized Mutual Information (NMI):** Measures the mutual information between the true and predicted labelings, normalized to be between 0 and 1. Values close to 1 indicate high agreement. `sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred)`.
*   **Homogeneity, Completeness, V-measure:** These three related metrics assess different aspects of the agreement. Homogeneity is high if each cluster contains only members of a single true class. Completeness is high if all members of a given true class are assigned to the same cluster. V-measure is the harmonic mean of homogeneity and completeness. `sklearn.metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)`.

External metrics provide a more objective evaluation if ground truth is available, but this scenario often bridges into semi-supervised learning rather than pure unsupervised discovery. They are useful for benchmarking clustering algorithms on datasets with known classes or assessing if data-driven clusters recover scientifically meaningful categories.

```python
# --- Code Example: Evaluating Clustering using Silhouette Score and ARI ---
# Note: Requires scikit-learn installation.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("Evaluating Clustering Performance:")

# Generate blobs data with known true labels
X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=1)
X_scaled = StandardScaler().fit_transform(X_blobs)
print(f"\nGenerated blobs data with {len(np.unique(y_true))} true classes.")

# --- Apply K-Means with different k values ---
k_values = range(2, 7)
kmeans_models = {}
kmeans_labels = {}
silhouette_scores = []
ari_scores = []

print("\nRunning K-Means for different k and evaluating...")
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)
    labels = kmeans.fit_predict(X_scaled)
    kmeans_models[k] = kmeans
    kmeans_labels[k] = labels
    
    # Calculate Silhouette Score (Internal Metric)
    # Requires features X and predicted labels
    if k > 1: # Silhouette needs at least 2 clusters
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}: Silhouette Score = {score:.4f}")
    else:
        silhouette_scores.append(np.nan) # Undefined for k=1

    # Calculate Adjusted Rand Index (External Metric - comparing to y_true)
    ari = adjusted_rand_score(y_true, labels)
    ari_scores.append(ari)
    print(f"  k={k}: Adjusted Rand Index (ARI) = {ari:.4f}")

# --- Plot Evaluation Metrics vs k ---
print("\nGenerating evaluation plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Silhouette Score
ax1.plot(k_values, silhouette_scores, 'o-', label='Silhouette Score')
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Silhouette Score")
ax1.set_title("Silhouette Score vs k for K-Means")
ax1.grid(True); ax1.legend()
best_k_silhouette = k_values[np.nanargmax(silhouette_scores)]
ax1.axvline(best_k_silhouette, color='red', linestyle='--', label=f'Best k = {best_k_silhouette}')
ax1.legend()


# Plot Adjusted Rand Index
ax2.plot(k_values, ari_scores, 'o-', label='Adjusted Rand Index')
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Adjusted Rand Index")
ax2.set_title("ARI vs k for K-Means (vs True Labels)")
ax2.grid(True); ax2.legend()
best_k_ari = k_values[np.argmax(ari_scores)]
ax2.axvline(best_k_ari, color='red', linestyle='--', label=f'Best k = {best_k_ari}')
ax2.legend()


fig.tight_layout()
# plt.show()
print("Evaluation plots generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code evaluates K-Means clustering for different numbers of clusters (k).
# 1. It generates 'blobs' data where the true number of clusters (`y_true`) is known (4).
# 2. It runs K-Means for k from 2 to 6.
# 3. For each k, it calculates:
#    a. Silhouette Score: An internal metric (`silhouette_score(X_scaled, labels)`) 
#       measuring cluster cohesion and separation. Higher is better.
#    b. Adjusted Rand Index (ARI): An external metric (`adjusted_rand_score(y_true, labels)`) 
#       comparing the K-Means labels to the known true labels. Higher (closer to 1) is better.
# 4. It plots both Silhouette Score and ARI as a function of k. 
# The Silhouette Score plot helps choose the 'optimal' k based only on the data structure 
# (often looking for a peak, here likely at k=4). The ARI plot confirms that k=4 indeed 
# yields the clustering that best matches the ground truth labels. This demonstrates 
# using both internal and external metrics for evaluation and model selection (choosing k).
```

**Visualization** also plays a crucial role in evaluating clustering. Plotting the data points colored by their assigned cluster labels (often after applying dimensionality reduction like PCA or UMAP if the data has more than 2 features, Sec 23.4/23.5) provides an immediate visual assessment of whether the clusters seem reasonable, compact, well-separated, or if the algorithm failed to capture the perceived structure. Comparing this visualization with plots using internal metrics (like plotting silhouette scores per point) or external labels can yield significant insights.

Ultimately, the "best" clustering is often defined by its utility for the specific scientific goal. A clustering solution might have a lower Silhouette Score but might separate data into groups that are more physically meaningful or interpretable in the context of the astronomical problem being addressed. Therefore, combining quantitative internal/external metrics with careful visual inspection and scientific interpretation is usually necessary for robust evaluation of clustering results.

**23.4 Dimensionality Reduction: Principal Component Analysis (PCA)**

High-dimensional datasets, common in astrophysics (e.g., spectra with thousands of wavelength bins, catalogs with dozens of measured parameters), pose challenges for both visualization and analysis. It's difficult to visualize data beyond 2 or 3 dimensions, and many machine learning algorithms can suffer from the "curse of dimensionality" – performance degrades, distances become less meaningful, and the required amount of training data increases exponentially as the number of features grows. **Dimensionality reduction** techniques aim to address this by transforming the data from a high-dimensional feature space into a lower-dimensional space while retaining as much of the relevant information or structure as possible. **Principal Component Analysis (PCA)** is the most widely used technique for **linear** dimensionality reduction.

PCA works by finding a new set of orthogonal (uncorrelated) axes, called **principal components (PCs)**, that capture the maximum variance present in the original data. The first principal component (PC1) is the direction in the feature space along which the data varies the most. The second principal component (PC2) is the direction orthogonal to PC1 that captures the largest *remaining* variance. Subsequent PCs capture progressively less variance, while remaining orthogonal to all preceding components. The total number of principal components is equal to the original number of features.

The transformation involves finding the eigenvectors and eigenvalues of the **covariance matrix** of the feature data `X`. The eigenvectors represent the directions of the principal components, and the corresponding eigenvalues represent the amount of variance captured by each component. By selecting only the first `k` eigenvectors corresponding to the `k` largest eigenvalues (where `k` is the desired lower dimensionality, `k < p` original features), we define a projection matrix. Multiplying the original data matrix `X` (after centering it by subtracting the mean of each feature) by this projection matrix transforms the data into the lower-dimensional space spanned by the top `k` principal components.

`scikit-learn.decomposition.PCA` provides an efficient implementation. Key steps involve:
1.  **Scaling Data:** PCA is sensitive to the relative scales of the input features because it's based on variance. Features with larger variances will dominate the principal components. Therefore, it is **essential** to **standardize** the data (using `StandardScaler` to give each feature zero mean and unit variance, Sec 20.2) *before* applying PCA.
2.  **Instantiation:** Create a `PCA` object, typically specifying the desired number of components `n_components`. If `n_components` is an integer `k`, it keeps the top `k` components. If it's a float between 0 and 1 (e.g., `0.95`), it automatically selects the minimum number of components required to retain that fraction of the total variance. If omitted, it keeps all components. `pca = PCA(n_components=...)`.
3.  **Fitting:** Fit the PCA model to the *scaled* training data: `pca.fit(X_train_scaled)`. This step calculates the principal components (eigenvectors) and their corresponding variances (eigenvalues) from the training data's covariance matrix.
4.  **Transformation:** Transform both the scaled training data and the scaled test data into the lower-dimensional principal component space using `pca.transform()`: `X_train_pca = pca.transform(X_train_scaled)`, `X_test_pca = pca.transform(X_test_scaled)`. The resulting arrays will have shape `(n_samples, n_components)`.

After fitting, the `PCA` object stores useful information:
*   `pca.explained_variance_ratio_`: An array showing the fraction of the total variance captured by each principal component, sorted in descending order. Plotting this "scree plot" helps visualize how much information is retained as more components are added and can guide the choice of `n_components`.
*   `pca.components_`: An array where each row represents a principal component (eigenvector) in terms of the original feature space. These components show the linear combination of original features that defines each PC, providing insight into what physical variations the PCs represent.
*   `pca.mean_`: The mean values subtracted during centering.
*   `pca.inverse_transform()`: Can approximately reconstruct the data back into the original feature space from the reduced PCA space (useful for visualization or assessing information loss).

PCA is valuable for several purposes:
*   **Visualization:** By reducing high-dimensional data to 2 or 3 principal components (`n_components=2` or `3`), we can create scatter plots (PC1 vs PC2, PC1 vs PC3, etc.) to visually explore the data structure, potentially revealing clusters or trends not obvious in individual feature plots.
*   **Noise Reduction:** Principal components associated with very small eigenvalues (low variance) often represent noise rather than significant signal structure. Discarding these components can sometimes act as a form of noise filtering.
*   **Feature Extraction / Compression:** Using the first few principal components (which capture most of the variance) as input features for subsequent supervised learning models (instead of all original features) can reduce dimensionality, potentially speed up training, and sometimes improve performance by removing noise and multicollinearity.

However, PCA has limitations. It assumes linear relationships and finds directions of maximum *variance*, which might not always correspond to the directions that best *separate* classes (for classification) or correlate most strongly with a target variable (for regression). The principal components themselves are linear combinations of original features and might not always have a clear physical interpretation, although examining the `pca.components_` vectors can provide clues. Furthermore, PCA is sensitive to feature scaling, making standardization essential beforehand. For capturing highly non-linear structures in data, non-linear dimensionality reduction techniques (Sec 23.5) are often more effective, especially for visualization.

```python
# --- Code Example: Applying PCA for Dimensionality Reduction & Visualization ---
# Note: Requires scikit-learn installation.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits # Example high-dimensional dataset

print("Applying Principal Component Analysis (PCA):")

# --- Load Example Data (Digits dataset: 64 features per image) ---
digits = load_digits()
X = digits.data # Shape (n_samples, 64 features)
y = digits.target # Target labels (digit 0-9)
print(f"\nLoaded Digits dataset: X shape={X.shape}, y shape={y.shape}")

# --- Step 1: Scale Data ---
print("Scaling data using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 2 & 3: Instantiate and Fit PCA ---
# Choose number of components (e.g., 2 for visualization, or based on variance)
n_comp = 2 
# Or use explained variance: pca = PCA(n_components=0.95) 
pca = PCA(n_components=n_comp)
print(f"\nFitting PCA with n_components={n_comp}...")
# Fit PCA on the scaled data
pca.fit(X_scaled) 
print("PCA fitting complete.")

# Examine explained variance
print("\nExplained Variance:")
print(f"  Variance ratio per component: {pca.explained_variance_ratio_}")
print(f"  Cumulative variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
print(f"  Total variance explained by {n_comp} components: {np.sum(pca.explained_variance_ratio_):.3f}")

# --- Step 4: Transform Data ---
print(f"\nTransforming data to {n_comp} PCA dimensions...")
X_pca = pca.transform(X_scaled)
print(f"  Transformed data shape: {X_pca.shape}") # Should be (n_samples, n_comp)

# --- Visualize Results (PCA Projection) ---
print("\nGenerating scatter plot of PCA components...")
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', # Color by true digit label
                     alpha=0.7, s=10)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title(f"PCA Projection of Digits Dataset (First 2 Components)")
ax.grid(True, alpha=0.4)
# Add legend for colors
legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
ax.add_artist(legend1)
fig.tight_layout()
# plt.show()
print("PCA plot generated.")
plt.close(fig)

# --- Examine Components (Eigenvectors) ---
# print("\nPrincipal Components (Eigenvectors):")
# print(pca.components_) # Shape (n_components, n_features)

print("-" * 20)

# Explanation: This code applies PCA to the high-dimensional (64 features) Digits dataset.
# 1. It loads the data and applies `StandardScaler` (crucial for PCA).
# 2. It instantiates `PCA`, requesting the top `n_components=2` components (suitable for 2D visualization).
# 3. It fits PCA to the scaled data using `pca.fit()`. This computes the principal axes.
# 4. It prints the `explained_variance_ratio_` for each component, showing how much 
#    of the total data variance is captured by PC1 and PC2.
# 5. It transforms the scaled data into the 2D PCA space using `pca.transform()`, 
#    resulting in `X_pca` with shape (n_samples, 2).
# 6. It creates a scatter plot of the data projected onto the first two principal 
#    components (PC1 vs PC2), coloring the points by their true digit label (`y`). 
#    This visualization shows how well PCA separates the different digits in a 
#    lower-dimensional space based on directions of maximum variance.
# (Commented code shows how to access the components/eigenvectors themselves).
```

In summary, PCA is a fundamental linear dimensionality reduction technique that finds orthogonal directions of maximum variance in the data. After standardizing features, `sklearn.decomposition.PCA` allows fitting the model, transforming data to a lower-dimensional space, and analyzing the explained variance. It's valuable for visualization, noise reduction, and creating compressed feature sets for other algorithms, but its linearity means it may not capture complex non-linear structures as effectively as manifold learning techniques.

**23.5 Manifold Learning (t-SNE, UMAP) for Visualization**

While Principal Component Analysis (PCA) provides a powerful tool for linear dimensionality reduction, many high-dimensional datasets in astrophysics exhibit complex, **non-linear structures** that PCA cannot effectively capture or represent in a low-dimensional visualization. For instance, galaxies might lie along a non-linear sequence in spectral feature space, or stellar populations might form curved manifolds in kinematic or chemical abundance space. **Manifold learning** encompasses a set of non-linear dimensionality reduction techniques designed specifically to uncover and visualize the underlying low-dimensional manifold (a smooth, lower-dimensional surface or curve) on which high-dimensional data points are assumed to lie. Two particularly popular and effective manifold learning algorithms for visualization are **t-Distributed Stochastic Neighbor Embedding (t-SNE)** and **Uniform Manifold Approximation and Projection (UMAP)**.

The primary goal of techniques like t-SNE and UMAP is **visualization**, typically projecting high-dimensional data down to 2 or 3 dimensions suitable for scatter plotting, in a way that preserves the *local structure* or *neighborhood relationships* of the data points. Unlike PCA which focuses on capturing global variance, these methods prioritize keeping points that are close together in the high-dimensional space close together in the low-dimensional map, potentially revealing clusters and manifold structures more clearly than PCA.

**t-SNE** (`sklearn.manifold.TSNE`) works by converting high-dimensional Euclidean distances between data points into conditional probabilities representing similarities. Specifically, it models the probability that point `i` would pick point `j` as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered on `i`. It then defines a similar probability distribution over pairs of points in the low-dimensional map (typically 2D or 3D), using a heavy-tailed Student's t-distribution (with one degree of freedom, resembling a Cauchy distribution). The algorithm then iteratively adjusts the positions of the points in the low-dimensional map to minimize the Kullback-Leibler (KL) divergence between the two distributions of pairwise similarities (high-dimensional vs. low-dimensional). This optimization encourages points that are similar (close) in high dimensions to be mapped to nearby points with high probability in the low-dimensional space.

Key characteristics and parameters of t-SNE:
*   **Excellent at revealing local structure and clusters:** Often produces visually compelling separations between groups present in the high-dimensional data.
*   **Non-deterministic:** The algorithm involves random initialization and optimization, meaning different runs might produce slightly different embeddings (though usually qualitatively similar). Using a fixed `random_state` is crucial for reproducibility.
*   **Computationally intensive:** Standard t-SNE scales poorly with the number of samples (O(n²)), making it slow for very large datasets (though faster approximations exist).
*   **Preserves local structure, not necessarily global distances:** The distances between separated clusters in the t-SNE plot are not necessarily meaningful. The primary focus is on keeping neighbors together.
*   **Sensitive to hyperparameters:** The `perplexity` parameter (typically between 5 and 50) relates to the number of effective nearest neighbors considered for each point and significantly influences the resulting visualization. Other parameters like `learning_rate` and `n_iter` also affect convergence. Tuning `perplexity` is often required.
*   **Feature Scaling:** Like most distance-based methods, applying feature scaling (e.g., `StandardScaler`) before t-SNE is generally recommended.

**UMAP (Uniform Manifold Approximation and Projection)** (`pip install umap-learn`) is a more recent manifold learning technique that has gained significant popularity due to its speed, scalability, and often superior preservation of both local and global data structure compared to t-SNE. UMAP is based on concepts from Riemannian geometry and algebraic topology. It constructs a high-dimensional graph representing the data's neighborhood structure, then optimizes a low-dimensional graph embedding to be as structurally similar as possible. It aims to model the data's underlying manifold and project it faithfully.

Key characteristics and parameters of UMAP:
*   **Fast and Scalable:** Generally much faster than t-SNE, capable of handling larger datasets.
*   **Better Global Structure Preservation:** Often does a better job than t-SNE at representing the larger-scale relationships and distances between clusters, in addition to preserving local neighborhoods.
*   **Deterministic (by default):** For a fixed `random_state`, UMAP typically yields the same embedding each time.
*   **Key Hyperparameters:**
    *   `n_neighbors`: Controls the balance between local (small `n_neighbors`) and global (large `n_neighbors`) structure preservation. Similar in spirit to t-SNE's perplexity, typical values range from 5 to 50.
    *   `min_dist`: Controls how tightly packed points are allowed to be in the low-dimensional embedding. Smaller values create more compact clusters, larger values spread points out more.
    *   `n_components`: The target dimensionality (usually 2 or 3 for visualization).
    *   `metric`: The distance metric used in the high-dimensional space (default 'euclidean', but others like 'cosine', 'manhattan' are possible).
*   **Feature Scaling:** Scaling is generally recommended before applying UMAP.

```python
# --- Code Example: Applying t-SNE and UMAP for Visualization ---
# Note: Requires scikit-learn, umap-learn (pip install umap-learn)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    import umap # umap-learn package
    umap_installed = True
except ImportError:
    umap_installed = False
    print("NOTE: 'umap-learn' not installed. Skipping UMAP example.")

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits # Use digits data again

print("Applying t-SNE and UMAP for visualizing high-dimensional data:")

# --- Load and Scale Data ---
digits = load_digits()
X = digits.data 
y = digits.target 
print(f"\nLoaded Digits dataset: X shape={X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled.")

# --- Apply t-SNE ---
# Usually apply to smaller subset if data is very large due to computation cost
n_subset = min(1000, len(X_scaled)) # Use subset for t-SNE speed if needed
subset_indices = np.random.choice(len(X_scaled), n_subset, replace=False)
X_subset = X_scaled[subset_indices]
y_subset = y[subset_indices]

print(f"\nApplying t-SNE (on {n_subset} samples)...")
# Common parameters: n_components=2, perplexity=30, random_state
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
            init='pca', n_iter=1000, random_state=42) 
X_tsne = tsne.fit_transform(X_subset) 
print("t-SNE transformation complete.")

# --- Apply UMAP (if installed) ---
X_umap = None
if umap_installed:
    print(f"\nApplying UMAP (on full {len(X_scaled)} samples)...")
    # Common parameters: n_neighbors=15, min_dist=0.1, n_components=2, random_state
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    print("UMAP transformation complete.")

# --- Visualize Results ---
print("\nGenerating visualization plots...")
n_plots = 1 + umap_installed
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
if n_plots == 1: axes = [axes] # Make iterable

# Plot t-SNE results
scatter_tsne = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='viridis', s=10, alpha=0.7)
axes[0].set_xlabel("t-SNE Dimension 1")
axes[0].set_ylabel("t-SNE Dimension 2")
axes[0].set_title(f"t-SNE Projection (Perplexity=30)")
axes[0].grid(True, alpha=0.4)
# Add legend
legend_tsne = axes[0].legend(*scatter_tsne.legend_elements(num=10), title="Digits", loc='center left', bbox_to_anchor=(1.05, 0.5))
axes[0].add_artist(legend_tsne)


# Plot UMAP results (if available)
if umap_installed and X_umap is not None:
    scatter_umap = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")
    axes[1].set_title(f"UMAP Projection (n_neighbors=15)")
    axes[1].grid(True, alpha=0.4)
    # Add legend
    legend_umap = axes[1].legend(*scatter_umap.legend_elements(num=10), title="Digits", loc='center left', bbox_to_anchor=(1.05, 0.5))
    axes[1].add_artist(legend_umap)

fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for legends outside
# plt.show()
print("Plots generated.")
plt.close(fig)
print("-" * 20)

# Explanation: This code applies both t-SNE and UMAP to the scaled Digits dataset.
# 1. It loads and scales the 64-dimensional digits data.
# 2. t-SNE: It applies `sklearn.manifold.TSNE` (potentially to a subset for speed) 
#    to get a 2D embedding `X_tsne`. Key parameters like `perplexity` are set. 
#    `init='pca'` and a fixed `random_state` are good practices.
# 3. UMAP: If the `umap-learn` package is installed, it applies `umap.UMAP` to the 
#    full scaled dataset to get a 2D embedding `X_umap`. Key parameters `n_neighbors` 
#    and `min_dist` control the embedding structure.
# 4. Visualization: It generates scatter plots of the 2D embeddings from both t-SNE 
#    and UMAP, coloring the points by their true digit label (`y`). These plots 
#    visually demonstrate how well each technique separates the different digit 
#    classes in the low-dimensional space based on the original high-dimensional 
#    features. Both often show clear clustering of digits, potentially better 
#    than PCA (shown in Sec 23.4) for this type of data. Legends are added outside 
#    the plot area for clarity.
```

Both t-SNE and UMAP are primarily **visualization techniques**. While the low-dimensional coordinates they produce can sometimes be used as features for subsequent clustering or classification, this should be done with caution, as the algorithms are optimized for visual separation rather than preserving information optimally for other tasks. The absolute coordinates and distances in t-SNE/UMAP plots are generally less meaningful than the relative groupings and neighborhood structures they reveal. They are powerful tools for gaining intuition about the structure of complex, high-dimensional astrophysical datasets, often revealing patterns invisible to linear methods like PCA. Choosing between them often involves trying both and selecting the one that provides the most insightful visualization for the specific data and scientific question, with UMAP often being preferred currently due to its speed and better global structure preservation.

**Application 23.A: Discovering Co-moving Star Groups with DBSCAN**

**Objective:** This application demonstrates using a density-based clustering algorithm, DBSCAN (Sec 23.2), to identify potential co-moving groups or open cluster remnants in a high-dimensional phase space derived from Gaia data. It involves preprocessing kinematic data (positions and velocities), applying DBSCAN, and visualizing the results. Reinforces Sec 23.2, 20.2 (scaling), 23.3 (evaluation concept), potentially 23.4/23.5 (visualization).

**Astrophysical Context:** Star clusters and associations are born together from the same molecular cloud, sharing similar ages, chemical compositions, and initial kinematics. While massive globular clusters remain gravitationally bound for billions of years, less massive open clusters and associations gradually disperse due to internal dynamics and Galactic tides. Identifying these dissolving or dispersed groups ("co-moving groups") relies on finding stars that still share similar velocities and occupy a relatively compact region of phase space (positions + velocities), even if they are spread across a wide area on the sky. Unsupervised clustering algorithms operating in the 6D phase space provided by Gaia (3D position + 3D velocity) are powerful tools for discovering such groups without prior knowledge of their existence or location.

**Data Source:** A catalog (`gaia_kinematics.csv` or FITS table) derived from Gaia data (e.g., Gaia DR3) for stars within a specific volume around the Sun (e.g., within 500 pc). Essential columns include 3D Cartesian positions (X, Y, Z, typically in pc, calculated from RA, Dec, parallax) and 3D Cartesian velocities (U, V, W, typically in km/s, calculated from RA, Dec, parallax, proper motions, and radial velocity) relative to a standard reference frame (e.g., Galactocentric or LSR). Radial velocities might be missing for some stars, requiring imputation (Sec 20.1) or limiting the analysis to 5D (X, Y, Z, pmra, pmdec). We will simulate 6D data for this example.

**Modules Used:** `pandas` or `astropy.table.Table` (for data), `numpy`, `sklearn.preprocessing.StandardScaler` (crucial for DBSCAN), `sklearn.cluster.DBSCAN`, `matplotlib.pyplot`, potentially `sklearn.metrics.silhouette_score` (for evaluation), `sklearn.decomposition.PCA` (for visualization).

**Technique Focus:** Preparing high-dimensional kinematic data for clustering. Applying `StandardScaler` is essential because DBSCAN relies on the distance metric `eps`, and features like positions (pc) and velocities (km/s) have vastly different scales. Using `DBSCAN` to find clusters based on density in the scaled 6D phase space. Tuning `eps` and `min_samples` parameters. Identifying clustered stars versus noise points (outliers). Visualizing the results, potentially using PCA to project the 6D space down to 2D for plotting, coloring points by cluster label.

**Processing Step 1: Load and Prepare Data:** Load the Gaia kinematic data (X, Y, Z, U, V, W) into a Pandas DataFrame or Astropy Table. Handle any missing values (e.g., impute missing radial velocities if calculating U,V,W, or work in 5D). Select the relevant 6 feature columns for clustering.

**Processing Step 2: Scale Features:** Create a `StandardScaler` object. Fit and transform the 6D feature matrix `X` to produce `X_scaled`, ensuring each feature has zero mean and unit variance. This makes the `eps` parameter in DBSCAN meaningful across all dimensions.

**Processing Step 3: Apply DBSCAN:** Instantiate `DBSCAN`, choosing appropriate values for `eps` and `min_samples`. These parameters are crucial and often require experimentation. `eps` defines the neighborhood size in the scaled 6D space; smaller values find denser clusters. `min_samples` defines the minimum number of points required to form a core point; higher values make the algorithm less sensitive to noise but might miss smaller groups. Typical starting points might involve exploring `eps` values based on nearest neighbor distances and setting `min_samples` based on the expected minimum size of a meaningful group (e.g., 5-10 stars). Run DBSCAN: `cluster_labels = dbscan.fit_predict(X_scaled)`. The result is an array where each element is the cluster label (0, 1, 2, ...) assigned to the corresponding star, or -1 if the star is classified as noise.

**Processing Step 4: Analyze Results:** Count the number of unique cluster labels found (excluding -1) and the number of noise points. Calculate internal validation metrics like the Silhouette Score (`silhouette_score(X_scaled, cluster_labels)`) if meaningful clusters are found (note: silhouette score might be low if many noise points exist or clusters are non-convex, and requires at least 2 clusters).

**Processing Step 5: Visualization:** Since visualizing 6D space is impossible, use dimensionality reduction for plotting. Apply PCA (Sec 23.4) to `X_scaled` to get the first two principal components (`X_pca`). Create a scatter plot of PC2 vs PC1, coloring each point according to its `cluster_labels`. This projection often reveals the separation of the clusters found by DBSCAN in the original high-dimensional space. Also plotting pairs of original kinematic variables (e.g., U vs V, X vs Y) colored by cluster label can provide physical intuition about the found groups.

**Output, Testing, and Extension:** Output includes the number of clusters found, the number of noise points, potentially the Silhouette Score, and visualization plots (e.g., PCA projection colored by cluster). **Testing:** Experiment with different `eps` and `min_samples` values and observe how the number and size of clusters change. Visually inspect the clusters in different projections (PCA, U vs V, X vs Y) – do they appear cohesive? Check if known clusters in the region (if any) are recovered. **Extensions:** (1) Analyze the properties (e.g., photometry, age estimates if available) of stars within the discovered clusters – do they share common characteristics? (2) Compare DBSCAN results with other algorithms like K-Means (after choosing k) or Hierarchical Clustering. (3) Use more sophisticated methods for choosing `eps` (e.g., analyzing the k-distance graph). (4) Incorporate measurement uncertainties into the clustering process (more advanced).

```python
# --- Code Example: Application 23.A ---
# Note: Requires scikit-learn, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs # Simulate clustered + noise data

print("Discovering Co-moving Groups with DBSCAN:")

# Step 1: Simulate 6D Phase Space Data (Simplified)
np.random.seed(101)
n_features = 6 # X, Y, Z, U, V, W
# Simulate two clusters in 6D space + background noise
cluster1_center = np.array([100, 50, 0, -10, -20, 5])
cluster2_center = np.array([150, -50, 20, 30, -10, -5])
cluster_std = 0.5 # Tighter clusters in scaled units for demo

X1, _ = make_blobs(n_samples=50, centers=[cluster1_center], cluster_std=cluster_std, n_features=n_features, random_state=0)
X2, _ = make_blobs(n_samples=40, centers=[cluster2_center], cluster_std=cluster_std, n_features=n_features, random_state=1)
# Add random background noise points
X_noise = np.random.uniform(X1.min(), X1.max(), size=(150, n_features)) * 2.0 # Wider spread

X = np.vstack((X1, X2, X_noise))
# True labels for verification later (0=cluster1, 1=cluster2, -1=noise - not used by DBSCAN)
y_true = np.concatenate((np.zeros(len(X1)), np.ones(len(X2)), -1 * np.ones(len(X_noise))))

print(f"\nGenerated {len(X)} simulated 6D data points (2 clusters + noise).")

# Step 2: Scale Features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply DBSCAN
# These parameters need tuning for real data!
eps_val = 0.5 
min_samples_val = 5 
print(f"\nApplying DBSCAN (eps={eps_val}, min_samples={min_samples_val})...")
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
cluster_labels = dbscan.fit_predict(X_scaled)

# Step 4: Analyze Results
# Core samples indices_ attribute stores indices of core points
# labels_ attribute stores cluster labels (-1 for noise)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
if hasattr(dbscan, 'core_sample_indices_'): # Check attribute exists
    core_samples_mask[dbscan.core_sample_indices_] = True

n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise_ = list(cluster_labels).count(-1)
print(f"\nEstimated number of clusters: {n_clusters_}")
print(f"Estimated number of noise points: {n_noise_}")

# Calculate Silhouette Score (only on non-noise points if clusters found)
if n_clusters_ > 1:
    try:
         labels_no_noise = cluster_labels[cluster_labels != -1]
         X_scaled_no_noise = X_scaled[cluster_labels != -1]
         silhouette_avg = silhouette_score(X_scaled_no_noise, labels_no_noise)
         print(f"Silhouette Coefficient (excluding noise): {silhouette_avg:.3f}")
    except ValueError:
         print("Could not calculate Silhouette Score (need > 1 cluster).")
else:
    print("Silhouette Score requires at least 2 clusters.")

# Step 5: Visualization (using PCA)
print("\nVisualizing using PCA projection...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled) # Reduce scaled data to 2D

fig, ax = plt.subplots(figsize=(8, 6))
# Use unique labels found by DBSCAN
unique_labels = set(cluster_labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        label_k = 'Noise'
        markersize = 3
    else:
        label_k = f'Cluster {k}'
        markersize = 6

    class_member_mask = (cluster_labels == k)
    
    # Plot core samples differently? (Optional)
    xy = X_pca[class_member_mask] # & core_samples_mask]
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), 
            markeredgecolor='k', markersize=markersize, label=label_k, alpha=0.7)
    
    # Plot non-core samples (if needed, for border points)
    # xy_noise = X_pca[class_member_mask & ~core_samples_mask]
    # ax.plot(xy_noise[:, 0], xy_noise[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=markersize // 2, alpha=0.5)


ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title(f"DBSCAN Clustering Results (PCA Projection)")
ax.legend()
ax.grid(True, alpha=0.4)
fig.tight_layout()
# plt.show()
print("Plot generated.")
plt.close(fig)

print("-" * 20)
```

**Application 23.B: Visualizing Galaxy Spectral Features with PCA/UMAP**

**Objective:** This application demonstrates using dimensionality reduction techniques – specifically PCA (Sec 23.4) and potentially UMAP (Sec 23.5) – to visualize the dominant patterns of variation within a large sample of high-dimensional galaxy spectra, allowing exploration of relationships between spectral shapes and physical properties.

**Astrophysical Context:** Galaxy spectra contain a wealth of information encoded in their continuum shape and the strengths of numerous absorption and emission lines. These features reflect the galaxy's stellar population (age, metallicity), star formation history, dust content, gas properties, and potential AGN activity. However, a single spectrum often consists of thousands of flux measurements (features). Dimensionality reduction provides a way to compress this information into a few key components that capture the main spectral variations across a large galaxy sample, facilitating visualization and classification.

**Data Source:** A collection of galaxy spectra (`galaxy_spectra.npy` or similar), pre-processed to a common rest-frame wavelength grid and potentially normalized (e.g., by median flux). Each spectrum is a 1D array of flux values. The collection forms a 2D array `X` where rows are galaxies and columns are flux bins (e.g., `[n_galaxies, n_wavelength_bins]`). Optionally, associated physical properties (like morphology, color, star formation rate) might be available for coloring the final visualization (`galaxy_properties.csv`).

**Modules Used:** `numpy` (for array manipulation), `sklearn.preprocessing.StandardScaler` (crucial for PCA/UMAP), `sklearn.decomposition.PCA`, `umap-learn` (for UMAP, if installed), `matplotlib.pyplot` (for plotting).

**Technique Focus:** Applying `StandardScaler` to spectral data (scaling each wavelength bin across the sample). Using `PCA` to find principal components (eigenspectra) and projecting the data onto the first few PCs. Analyzing the explained variance ratio to determine how many PCs capture most information. Optionally using `UMAP` for non-linear projection to 2D. Visualizing the galaxy sample in the reduced PCA or UMAP space, potentially coloring points by known physical properties to see if the low-dimensional embedding correlates with physical characteristics.

**Processing Step 1: Load and Prepare Data:** Load the spectral data matrix `X [n_galaxies, n_wavelengths]`. Load associated properties `props` if available. Ensure data is clean (e.g., handle bad pixels/wavelengths).

**Processing Step 2: Scale Features:** Since PCA finds directions of maximum *variance*, scaling is crucial if different wavelength bins have vastly different flux variances (which they often do). Apply `StandardScaler` to `X` *across the sample axis* (i.e., scale each column/wavelength bin independently): `X_scaled = StandardScaler().fit_transform(X)`.

**Processing Step 3: Apply PCA:** Instantiate `PCA`, perhaps requesting a reasonable number of components (e.g., `n_components=10`) or enough to capture a high fraction of variance (e.g., `n_components=0.99`). Fit PCA on the scaled data: `pca = PCA(...); pca.fit(X_scaled)`. Transform the data: `X_pca = pca.transform(X_scaled)`. Examine `pca.explained_variance_ratio_` (e.g., via a scree plot) to see how many components are significant.

**Processing Step 4 (Optional): Apply UMAP:** If desired for non-linear visualization, apply UMAP to the scaled data `X_scaled` or potentially to the first few significant PCA components `X_pca[:, :k]` (applying UMAP after PCA can sometimes be faster and less noisy): `reducer = umap.UMAP(n_components=2, ...); X_umap = reducer.fit_transform(X_relevant)`.

**Processing Step 5: Visualization:**
    *   Plot the first few principal component vectors (`pca.components_[:num]`) as spectra (eigen-spectra) vs. wavelength to try and interpret the main modes of variation they represent (e.g., PC1 might represent continuum slope/stellar age, PC2 emission line strength/SFR).
    *   Create scatter plots of the galaxy sample projected onto the first few PCA components (PC1 vs PC2, PC1 vs PC3).
    *   If UMAP was run, create a scatter plot of UMAP Dim 1 vs UMAP Dim 2.
    *   Color the points in the scatter plots by a known physical property (e.g., g-r color, star formation rate, morphology) to see if the dimensionality reduction technique successfully groups or orders galaxies according to these properties in the low-dimensional space.

**Output, Testing, and Extension:** Output includes the scree plot of explained variance, plots of the first few eigen-spectra, and scatter plots of the data in the reduced PCA and/or UMAP space, potentially colored by physical properties. **Testing:** Verify the shape of the transformed data (`X_pca`, `X_umap`). Check if the cumulative explained variance reaches a high level with a reasonable number of PCs. Examine eigen-spectra for physically interpretable features. Check if the low-dimensional plots show meaningful separation or trends related to the coloring property. **Extensions:** (1) Use the PCA/UMAP coordinates as input features for a clustering algorithm (Chapter 23) to automatically find groups based on spectral shape. (2) Use the PCA/UMAP coordinates as input features for a supervised classifier or regressor (Chapters 21/22) to predict galaxy properties from the compressed spectral information. (3) Try different scaling methods (e.g., normalizing each spectrum individually before PCA). (4) Compare PCA with other linear methods like Non-negative Matrix Factorization (NMF) if spectra are non-negative.

```python
# --- Code Example: Application 23.B ---
# Note: Requires scikit-learn, umap-learn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    import umap
    umap_installed = True
except ImportError:
    umap_installed = False
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml # To get example spectral data (if possible)

print("Visualizing Galaxy Spectral Features with PCA/UMAP:")

# Step 1: Load/Simulate Spectral Data & Properties
print("\nLoading/Simulating Galaxy Spectra...")
# --- Option A: Try to load real data (e.g., SDSS spectra subset via OpenML) ---
X, y_props = None, None
try:
    # Example: Fetch a subset of SDSS spectra data from OpenML (may change/be unavailable)
    # data_bunch = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False) # Placeholder
    # Need a real spectral dataset here. For now, simulate.
    pass # Replace with actual data loading if available
except Exception as e:
    print(f"  Could not load real data, simulating instead: {e}")
    
if X is None: # If loading failed, simulate
    n_galaxies = 200
    n_wavelengths = 500
    wavelengths = np.linspace(4000, 8000, n_wavelengths) # Angstroms
    X = np.zeros((n_galaxies, n_wavelengths))
    props = {'color': np.zeros(n_galaxies), 'sfr': np.zeros(n_galaxies)}
    # Simulate two types of galaxies
    for i in range(n_galaxies):
        continuum_slope = np.random.normal(-1.0, 0.3)
        continuum = 10 * (wavelengths/6000)**continuum_slope
        if i < n_galaxies // 2: # Type 1: Older/Redder
             props['color'][i] = 1.2 + np.random.normal(0, 0.1)
             props['sfr'][i] = np.random.uniform(0, 0.1)
             # Add absorption lines (simplified)
             abs_line_idx = np.argmin(np.abs(wavelengths - 5175)) # Mg b
             continuum[abs_line_idx-5:abs_line_idx+5] *= np.random.uniform(0.7, 0.9)
        else: # Type 2: Younger/Bluer/SF
             props['color'][i] = 0.5 + np.random.normal(0, 0.1)
             props['sfr'][i] = np.random.uniform(0.5, 2.0)
             # Add emission line (simplified)
             em_line_idx = np.argmin(np.abs(wavelengths - 6563)) # Halpha
             continuum[em_line_idx-3:em_line_idx+3] += np.random.uniform(10, 30)
        X[i, :] = continuum + np.random.normal(0, 1.0, n_wavelengths) # Add noise
    y_props = pd.DataFrame(props)
    print(f"  Simulated {n_galaxies} spectra with {n_wavelengths} bins.")

# Step 2: Scale Features (each wavelength bin)
print("\nScaling spectral data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
n_pca_comp = 10 # Keep first 10 components initially
print(f"\nApplying PCA (n_components={n_pca_comp})...")
pca = PCA(n_components=n_pca_comp)
# Fit and transform in one step
X_pca = pca.fit_transform(X_scaled) 
print(f"  Transformed data shape (PCA): {X_pca.shape}")
print(f"  Explained variance ratio (first 5): {pca.explained_variance_ratio_[:5].round(3)}")
print(f"  Cumulative variance (first 5): {np.cumsum(pca.explained_variance_ratio_[:5]).round(3)}")

# Step 4 (Optional): Apply UMAP (on scaled data or first few PCs)
X_umap = None
if umap_installed:
    print(f"\nApplying UMAP (n_components=2)...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    # Apply on first few PCA components for speed/noise reduction
    # Or apply on X_scaled directly: X_umap = reducer.fit_transform(X_scaled)
    if X_pca.shape[1] >= 5:
         X_umap = reducer.fit_transform(X_pca[:, :5]) 
    else:
         X_umap = reducer.fit_transform(X_scaled) 
    print(f"  Transformed data shape (UMAP): {X_umap.shape}")

# Step 5: Visualization
print("\nGenerating plots...")
# Plot Explained Variance
fig_var, ax_var = plt.subplots(figsize=(6, 4))
ax_var.plot(np.arange(1, n_pca_comp + 1), np.cumsum(pca.explained_variance_ratio_), 'o-')
ax_var.set_xlabel("Number of Principal Components")
ax_var.set_ylabel("Cumulative Explained Variance Ratio")
ax_var.set_title("PCA Explained Variance")
ax_var.grid(True); ax_var.set_ylim(0, 1.05)
# plt.show()
print("  Explained variance plot generated.")
plt.close(fig_var)

# Plot Eigenspectra (first 3 PCs)
fig_eig, axes_eig = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
for i in range(3):
    axes_eig[i].plot(wavelengths, pca.components_[i, :])
    axes_eig[i].set_ylabel(f"PC {i+1}")
    axes_eig[i].grid(True)
axes_eig[-1].set_xlabel("Wavelength (Angstrom)")
fig_eig.suptitle("First 3 Principal Components (Eigenspectra)")
fig_eig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
print("  Eigenspectra plot generated.")
plt.close(fig_eig)

# Plot PCA/UMAP projections, colored by property (e.g., 'color')
plot_prop = 'color' # or 'sfr'
if plot_prop in y_props.columns:
    prop_values = y_props[plot_prop]
    norm = plt.Normalize(prop_values.min(), prop_values.max())
    cmap = 'viridis'
    
    n_plots_proj = 1 + (X_umap is not None)
    fig_proj, axes_proj = plt.subplots(1, n_plots_proj, figsize=(6.5 * n_plots_proj, 5.5), squeeze=False)
    
    # PCA Plot
    scatter_pca = axes_proj[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=prop_values, cmap=cmap, norm=norm, s=10, alpha=0.7)
    axes_proj[0, 0].set_xlabel("Principal Component 1")
    axes_proj[0, 0].set_ylabel("Principal Component 2")
    axes_proj[0, 0].set_title(f"PCA Projection colored by {plot_prop}")
    axes_proj[0, 0].grid(True, alpha=0.4)
    fig_proj.colorbar(scatter_pca, ax=axes_proj[0,0], label=plot_prop)
    
    # UMAP Plot
    if X_umap is not None:
        scatter_umap = axes_proj[0, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=prop_values, cmap=cmap, norm=norm, s=10, alpha=0.7)
        axes_proj[0, 1].set_xlabel("UMAP Dimension 1")
        axes_proj[0, 1].set_ylabel("UMAP Dimension 2")
        axes_proj[0, 1].set_title(f"UMAP Projection colored by {plot_prop}")
        axes_proj[0, 1].grid(True, alpha=0.4)
        fig_proj.colorbar(scatter_umap, ax=axes_proj[0,1], label=plot_prop)

    fig_proj.tight_layout()
    # plt.show()
    print("  Projection plots generated.")
    plt.close(fig_proj)
else:
     print("  Cannot color plots (property data not available).")

print("-" * 20)
```

**Summary**

This chapter explored the domain of **unsupervised learning**, focusing on techniques that discover inherent structure within unlabeled data, a common scenario in astrophysical exploration. The primary goals discussed were **clustering** – grouping similar data points – and **dimensionality reduction** – simplifying high-dimensional data while preserving key information, often for visualization or feature extraction. Several standard clustering algorithms implemented in `scikit-learn` were introduced: **K-Means**, a simple and efficient method partitioning data into a predefined number `k` of spherical clusters based on minimizing within-cluster variance; **DBSCAN**, a density-based approach that finds arbitrarily shaped clusters and identifies noise points without requiring `k` but sensitive to density parameters (`eps`, `min_samples`); and **Hierarchical Clustering** (agglomerative), which builds a nested hierarchy of clusters visualized as a dendrogram. The challenge of evaluating clustering performance without ground truth labels was addressed, introducing internal metrics like the **Silhouette Score** (measuring cohesion and separation) and external metrics (like Adjusted Rand Index, NMI) used when known labels are available for comparison.

The second major topic was dimensionality reduction. **Principal Component Analysis (PCA)**, a fundamental linear technique implemented in `sklearn.decomposition.PCA`, was detailed. PCA finds orthogonal axes (principal components) that capture the maximum variance in the data, enabling projection onto a lower-dimensional space. The importance of feature scaling (`StandardScaler`) before PCA was emphasized, along with interpreting the `explained_variance_ratio_` (scree plots) and the principal component vectors (eigenspectra/eigenimages). Recognizing PCA's limitations with non-linear structures, the chapter then introduced powerful non-linear manifold learning techniques primarily used for visualization: **t-Distributed Stochastic Neighbor Embedding (t-SNE)** (`sklearn.manifold.TSNE`), which excels at revealing local structure and clusters in 2D/3D but is computationally intensive and sensitive to parameters like `perplexity`; and **Uniform Manifold Approximation and Projection (UMAP)** (via the `umap-learn` library), a faster and often more effective alternative that aims to preserve both local and global data structure, controlled by parameters like `n_neighbors` and `min_dist`. The applications demonstrated using these techniques for finding co-moving star groups in Gaia data (DBSCAN) and visualizing galaxy spectral variations (PCA/UMAP), highlighting the power of unsupervised methods for discovery and exploration in complex astrophysical datasets.

---

**References for Further Reading:**

1.  **Ivezić, Ž., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).** *Statistics, Data Mining, and Machine Learning in Astronomy*. Princeton University Press. (Relevant chapters often available online, e.g., Chapter 6 covers dimensionality reduction, Chapter 7 covers clustering: [http://press.princeton.edu/titles/10159.html](http://press.princeton.edu/titles/10159.html))
    *(Provides excellent coverage of PCA, manifold learning (including t-SNE concepts), K-Means, hierarchical clustering, and DBSCAN with astronomical examples.)*

2.  **VanderPlas, J. (2016).** *Python Data Science Handbook: Essential Tools for Working with Data*. O'Reilly Media. (Chapter 5: Machine Learning, Sections on PCA, Manifold Learning, K-Means, DBSCAN available online: [https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html), [https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html](https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html), [https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html](https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html), [https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html) - GMM often used alongside clustering)
    *(Offers clear explanations and practical Python code examples for PCA, Manifold Learning (including t-SNE), K-Means, and DBSCAN using Scikit-learn.)*

3.  **The Scikit-learn Developers. (n.d.).** *Scikit-learn Documentation: User Guide*. Scikit-learn. Retrieved January 16, 2024, from [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html) (Specific sections on Clustering, Decomposition (PCA), Manifold learning (t-SNE), Metrics (evaluation))
    *(The essential resource for implementation details, API reference, and usage examples for `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `PCA`, `TSNE`, and evaluation metrics like `silhouette_score` discussed in this chapter.)*

4.  **McInnes, L., Healy, J., & Melville, J. (2018).** UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*. [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426) (See also `umap-learn` documentation: [https://umap-learn.readthedocs.io/en/latest/](https://umap-learn.readthedocs.io/en/latest/))
    *(The paper introducing the UMAP algorithm. The linked documentation is crucial for understanding its parameters and usage, relevant to Sec 23.5.)*

5.  **Van der Maaten, L., & Hinton, G. (2008).** Visualizing Data using t-SNE. *Journal of Machine Learning Research*, *9*, 2579-2605. ([Link via JMLR](https://www.jmlr.org/papers/v9/vandermaaten08a.html))
    *(The original paper describing the t-SNE algorithm, providing theoretical background for the technique discussed in Sec 23.5.)*
