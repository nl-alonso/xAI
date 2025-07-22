import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_boolean(X, y, k_neighbors=5, sampling_strategy='auto'):
    """
    Example Python function implementing a SMOTE variant for datasets with boolean features.
    SMOTE-N for nominal features, using Value Difference Metric (VDM) for distances.
    Assumes all features in X are boolean (0 or 1), and the target y is binary (0 or 1).
    Generates synthetic samples for the minority class to balance the dataset.
    Designed to be performant using NumPy vectorization and scikit-learn's NearestNeighbors.
    For large datasets, NearestNeighbors with custom metric may fall back to brute force, which is O(n^2),
    but is scalable for moderate sizes (n < 10k). For very large n, consider approximate NN methods.

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features), with values 0 or 1 (boolean features).
    - y: np.ndarray of shape (n_samples,), with values 0 or 1 (binary target).
    - k_neighbors: int, number of nearest neighbors to consider (default=5).
    - sampling_strategy: str or float, 'auto' to balance to majority size, or float as ratio to majority.

    Returns:
    - X_res: np.ndarray, resampled features including synthetics.
    - y_res: np.ndarray, resampled targets.
    """

    if not np.all(np.isin(X, [0, 1])):
        raise ValueError("All features in X must be boolean (0 or 1).")

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Target y must be binary (0 or 1).")

    # Compute class counts
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    n_majority = counts.max()
    n_minority = counts.min()

    # Determine number of synthetic samples to generate
    if sampling_strategy == 'auto':
        n_synth = n_majority - n_minority
    else:
        n_synth = int(sampling_strategy * n_majority) - n_minority

    if n_synth <= 0:
        return X.copy(), y.copy()

    # Get minority samples
    min_mask = (y == minority_class)
    min_X = X[min_mask]
    min_indices = np.arange(len(y))[min_mask]  # not used, but for clarity
    n_min = len(min_X)

    if n_min <= 1:
        # Too few minority samples, duplicate the existing ones
        synth_indices = np.random.choice(n_min, n_synth, replace=True)
        synth_X = min_X[synth_indices]
        synth_y = np.full(n_synth, minority_class)
    else:
        # Compute VDM deltas for each feature
        n_features = X.shape[1]
        deltas = np.zeros(n_features)
        for f in range(n_features):
            mask0 = (X[:, f] == 0)
            C0 = np.sum(mask0)
            C1 = len(y) - C0
            if C0 == 0 or C1 == 0:
                deltas[f] = 0.0  # No difference observable
            else:
                C00 = np.sum(y[mask0] == 0)
                C01 = np.sum(y[mask0] == 1)
                C10 = np.sum(y[~mask0] == 0)
                C11 = np.sum(y[~mask0] == 1)
                deltas[f] = abs(C00 / C0 - C10 / C1) + abs(C01 / C0 - C11 / C1)

        # Define custom VDM distance (weighted Hamming essentially)
        def vdm_dist(u, v):
            diff = (u != v)
            return np.sum(deltas[diff])

        # Adjust k if necessary
        k_used = min(k_neighbors, n_min - 1)

        # Find k nearest neighbors for each minority sample
        nn = NearestNeighbors(n_neighbors=k_used + 1, metric=vdm_dist)
        nn.fit(min_X)
        _, indices = nn.kneighbors(min_X)  # indices[:, 0] is self, [:, 1:] are neighbors

        # Generate synthetic samples
        synth_X = np.empty((n_synth, n_features), dtype=int)
        synth_y = np.full(n_synth, minority_class)
        rng = np.random.default_rng()
        for s in range(n_synth):
            # Select a random minority sample index
            i = rng.integers(0, n_min)
            # Get its k neighbors' indices in min_X
            neighbor_indices = indices[i, 1:]
            # Group: the sample + its k neighbors
            group_indices = np.concatenate(([i], neighbor_indices))
            group_values = min_X[group_indices]
            # For each feature, compute mode
            for f in range(n_features):
                vals, cnts = np.unique(group_values[:, f], return_counts=True)
                synth_X[s, f] = vals[np.argmax(cnts)]

    # Combine original and synthetic
    X_res = np.vstack((X, synth_X))
    y_res = np.concatenate((y, synth_y))

    return X_res, y_res