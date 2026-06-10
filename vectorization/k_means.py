import numpy as np


class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def _compute_distances(self, X, centroids):
        """
        Input:
            X: (N, D),
            centroids: (k, D)
        Return:
            distances: (N, k)

        """
        # ||x_i - c_j||^2 = ||x_i||^2 + ||c_j||^2 - 2 x_i^\top c_j
        x_sq = (X**2).sum(-1)[:, None]  # (N, 1)
        c_sq = (centroids**2).sum(-1)  # (k,)
        distances = x_sq + c_sq - 2 * X @ centroids.T
        return distances

    def _run_single_init(self, X, centroids):
        """
        Input:
            X: (N, D)
            centroids: (k, D)
        Output:
            centroids: (k, D)
            inertia: float
        """
        N, D = X.shape
        k, _ = centroids.shape

        for _ in range(self.max_iters):
            distances = self._compute_distances(X, centroids)  # (N, k)

            # Compute new centroids
            labels = np.argmin(distances, axis=-1)  # (N,)
            mask = labels[:, None] == np.arange(k)  # (N, k)
            new_centroids = (X[:, None, :] * mask[:, :, None]).sum(0)  # (k, D)
            counts = mask.sum(0)

            num_empty_cluster = np.sum(counts == 0)
            replacement = X[np.random.choice(N, (num_empty_cluster,))]
            new_centroids[counts == 0] = replacement

            counts = np.where(counts == 0, 1, counts)
            new_centroids /= counts[:, None]  # (k, D)

            # labels = np.argmin(distances, axis=-1)  # (N,)
            # mask = labels[:, None] == np.arange(k)  # (N, k)
            # cluster_sum = (X[:, None, :] * mask[:, :, None]).sum(0)  # (k, D)
            # empty_centroids = mask.sum(0) == 0  # (k,)
            # if np.sum(empty_centroids) > 0:
            #     cluster_sum[empty_centroids] = X[
            #         np.random.choice(N, (empty_centroids.sum()))
            #     ]
            # point_counts = np.where(mask.sum(0) == 0, 1, mask.sum(0))  # (k,)
            # new_centroids = cluster_sum / point_counts[:, None]

            # Check convergence criteria
            delta = np.linalg.norm(new_centroids - centroids, axis=-1).max()
            centroids = new_centroids
            if delta < self.tol:
                break

        # Compute inertia
        best_dist = self._compute_distances(X, centroids)
        inertia = np.sum(np.min(best_dist**2, axis=-1))

        return centroids, inertia

        # Optimize centroids until convergence
        # for _ in range(self.max_iters):
        #     distances = self._compute_distances(X, centroids)

        #     labels = np.argmin(distances, axis=1)

        #     new_centroids = np.zeros_like(centroids)
        #     for j in range(self.k):
        #         cluster_points = X[labels == j]

        #         if len(cluster_points) > 0:
        #             new_centroids[j] = cluster_points.mean(axis=0)
        #         else:
        #             new_centroids[j] = X[np.random.choice(X.shape[0])]

        #     shift = np.linalg.norm(centroids - new_centroids)
        #     centroids = new_centroids

        #     if shift < self.tol:
        #         break

        # # Inertia is the sum of squared distances to each closest centroid
        # x_sq = np.sum(X**2, axis=1)[:, np.newaxis]
        # c_sq = np.sum(centroids**2, axis=1)
        # final_dists_sq = np.maximum(0, x_sq + c_sq - 2 * np.dot(X, centroids.T))
        # inertia = np.sum(np.min(final_dists_sq, axis=1))

        # return centroids, inertia

    def fit(self, X, n_init=10):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_inertia = np.inf
        best_centroids = None

        # Run the algorithm n_init times to avoid local minima traps
        for _ in range(n_init):
            # Initialize safely
            if X.shape[0] >= self.k:
                initial_idx = np.random.choice(X.shape[0], self.k, replace=False)
            else:
                initial_idx = np.random.choice(X.shape[0], self.k, replace=True)

            initial_centroids = X[initial_idx].copy()
            centroids, inertia = self._run_single_init(X, initial_centroids)

            # Track the best global minimum across initializations
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()

        self.centroids = best_centroids
        return self

    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)


def test_kmeans():
    print("Running K-Means functional validation tests...")

    # Helper function to solve the permutation invariance problem
    def sort_centroids(centroids):
        # Sort lexicographically by columns (last column first, then second to last, etc.)
        return centroids[np.lexsort(centroids.T[::-1])]

    # --- Test 1: Convergence on distinct clusters ---
    # Create 3 tightly packed, widely separated clusters
    true_centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]])

    np.random.seed(42)
    # Generate 100 points around each center with small variance
    X_synthetic = np.vstack(
        [center + np.random.randn(100, 2) * 0.1 for center in true_centers]
    )

    # Shuffle the dataset
    np.random.shuffle(X_synthetic)

    kmeans = KMeans(k=3, random_state=42)
    kmeans.fit(X_synthetic)

    sorted_found_centroids = sort_centroids(kmeans.centroids)
    sorted_true_centers = sort_centroids(true_centers)

    np.testing.assert_allclose(
        sorted_found_centroids,
        sorted_true_centers,
        atol=0.05,
        err_msg="K-Means failed to converge to the correct true cluster centers.",
    )
    print("✓ Centroid Convergence: Algorithm accurately found true cluster centers.")

    # --- Test 2: Inference / Prediction Parity ---
    # Test points exactly on the true centers
    test_points = np.array([[10.0, 10.0], [0.0, 0.0]])
    labels = kmeans.predict(test_points)

    # The predicted centroids for these points should exactly match the test points
    predicted_centroids = kmeans.centroids[labels]
    np.testing.assert_allclose(
        predicted_centroids,
        test_points,
        atol=0.05,
        err_msg="Prediction failed to assign points to the obvious nearest centroid.",
    )
    print("✓ Inference Parity: Prediction mapped to the correct existing clusters.")

    # --- Test 3: Empty Cluster Edge Case Handling ---
    # Force an edge case by requesting 5 clusters for only 3 identical points
    X_edge = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    kmeans_edge = KMeans(k=5, max_iters=10, random_state=1)

    try:
        kmeans_edge.fit(X_edge)
        assert not np.isnan(kmeans_edge.centroids).any(), (
            "NaN values generated during empty cluster handling."
        )
        print("✓ Edge Case: Empty cluster handling executed without throwing NaNs.")
    except Exception as e:
        raise AssertionError(f"Edge case testing failed with exception: {e}")

    print("\nAll K-Means tests passed successfully.")


if __name__ == "__main__":
    test_kmeans()
