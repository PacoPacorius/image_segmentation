import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def n_cuts(affinity_mat, k):
    """
    Non-recursive normalized cuts method.
    
    Parameters:
    affinity_mat: numpy array with shape [n, n], affinity/weight matrix
    k: int, number of clusters
    
    Returns:
    cluster_idx: numpy array with shape [n,], cluster labels for each vertex
    """
    n = affinity_mat.shape[0]
    
    # Step 1: Calculate Laplacian matrix L = D - W
    W = affinity_mat
    degree = np.sum(W, axis=1)
    D = np.diag(degree)
    L = D - W
    
    # Step 2: Solve generalized eigenvalue equation Lx = λDx
    # This is equivalent to solving (D^-1 * L)x = λx
    # We need to handle the case where D might have zero diagonal elements
    D_inv = np.diag(1.0 / (degree))  # Add small epsilon to avoid division by zero
    L_normalized = D_inv @ L
    
    # Find k smallest eigenvalues and corresponding eigenvectors
    eigenvalues, eigenvectors = eigs(L_normalized, k=k, which='SM')
    
    # Sort eigenvalues and eigenvectors by eigenvalue magnitude
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 3: Create matrix U with eigenvectors as columns
    U = np.real(eigenvectors)  # Take real part in case of numerical issues
    
    # Step 4: Apply k-means clustering to rows of U
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    cluster_idx = kmeans.labels_
    
    return cluster_idx.astype(np.float64)

def calculate_n_cut_value(affinity_mat, cluster_idx):
    """
    Calculate the n_cut value for two clusters.
    
    Parameters:
    affinity_mat: numpy array with shape [n, n], affinity/weight matrix
    cluster_idx: numpy array with shape [n,], cluster labels (should have exactly 2 unique values)
    
    Returns:
    n_cut: float, the normalized cut value
    """
    W = affinity_mat
    unique_clusters = np.unique(cluster_idx)
    
    if len(unique_clusters) != 2:
        raise ValueError("calculate_n_cut_value expects exactly 2 clusters")
    
    # Get indices for clusters A and B
    cluster_A_indices = np.where(cluster_idx == unique_clusters[0])[0]
    cluster_B_indices = np.where(cluster_idx == unique_clusters[1])[0]
    
    # Calculate assoc(A,A) - sum of weights within cluster A
    assoc_A_A = np.sum(W[np.ix_(cluster_A_indices, cluster_A_indices)])
    
    # Calculate assoc(B,B) - sum of weights within cluster B
    assoc_B_B = np.sum(W[np.ix_(cluster_B_indices, cluster_B_indices)])
    
    # Calculate assoc(A,V) - sum of all weights from cluster A to all vertices
    assoc_A_V = np.sum(W[cluster_A_indices, :])
    
    # Calculate assoc(B,V) - sum of all weights from cluster B to all vertices
    assoc_B_V = np.sum(W[cluster_B_indices, :])
    
    # Calculate Nassoc(A,B)
    Nassoc_A_B = (assoc_A_A / (assoc_A_V + 1e-10)) + (assoc_B_B / (assoc_B_V + 1e-10))
    
    # Calculate Ncut(A,B)
    n_cut = 2.0 - Nassoc_A_B
    
    return n_cut

def n_cuts_recursive(affinity_mat, T1, T2):
    n = affinity_mat.shape[0]
    cluster_idx = np.zeros(n)
    current_label = 0

    def recurse(indices, label):
        # Base case: if the cluster is too small, assign the current label and stop splitting
        if len(indices) <= T1:
            cluster_idx[indices] = label
            return

        # Extract the submatrix of the affinity matrix for the current indices
        sub_aff = affinity_mat[np.ix_(indices, indices)]

        # Perform non-recursive normalized cuts with k=2
        sub_labels = n_cuts(sub_aff, 2)

        # Compute the n_cut value for the partition
        n_cut = calculate_n_cut_value(sub_aff, sub_labels)

        # If the cut is not good enough, stop splitting and assign label
        if n_cut > T2:
            cluster_idx[indices] = label
            return

        # Split the current indices into two based on clustering
        A = indices[sub_labels == 0]
        B = indices[sub_labels == 1]

        # Recursively split each subset
        recurse(A, label)  # Keep the same label for one subset
        recurse(B, max(cluster_idx) + 1)  # Assign new label to the other subset

    # Start recursion with all indices and the initial label
    recurse(np.arange(n), current_label)
    return cluster_idx

