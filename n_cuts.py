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
    eigenvalues, eigenvectors = eigs(L_normalized, k=k, which='SM', ncv=100)
    
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
    A_idx = np.where(cluster_idx == 0)[0]
    B_idx = np.where(cluster_idx == 1)[0]

    assoc_AA = np.sum(affinity_mat[np.ix_(A_idx, A_idx)])
    assoc_AV = np.sum(affinity_mat[A_idx])
    assoc_BB = np.sum(affinity_mat[np.ix_(B_idx, B_idx)])
    assoc_BV = np.sum(affinity_mat[B_idx])

    Nassoc = (assoc_AA / assoc_AV) + (assoc_BB / assoc_BV)
    return 2 - Nassoc

#def calculate_n_cut_value(affinity_mat, cluster_idx):
    #"""
    #Calculate the n_cut value for two clusters.
    #
    #Parameters:
    #affinity_mat: numpy array with shape [n, n], affinity/weight matrix
    #cluster_idx: numpy array with shape [n,], cluster labels (should have exactly 2 unique values)
    #
    #Returns:
    #n_cut: float, the normalized cut value
    #"""
    #W = affinity_mat
    #unique_clusters = np.unique(cluster_idx)
    #
    #if len(unique_clusters) != 2:
        #raise ValueError("calculate_n_cut_value expects exactly 2 clusters")
    #
    ## Get indices for clusters A and B
    #cluster_A_indices = np.where(cluster_idx == unique_clusters[0])[0]
    #cluster_B_indices = np.where(cluster_idx == unique_clusters[1])[0]
    #
    ## Calculate assoc(A,A) - sum of weights within cluster A
    #assoc_A_A = np.sum(W[np.ix_(cluster_A_indices, cluster_A_indices)])
    #
    ## Calculate assoc(B,B) - sum of weights within cluster B
    #assoc_B_B = np.sum(W[np.ix_(cluster_B_indices, cluster_B_indices)])
    #
    ## Calculate assoc(A,V) - sum of all weights from cluster A to all vertices
    #assoc_A_V = np.sum(W[cluster_A_indices, :])
    #
    ## Calculate assoc(B,V) - sum of all weights from cluster B to all vertices
    #assoc_B_V = np.sum(W[cluster_B_indices, :])
    #
    ## Calculate Nassoc(A,B)
    #Nassoc_A_B = (assoc_A_A / (assoc_A_V + 1e-10)) + (assoc_B_B / (assoc_B_V + 1e-10))
    #
    ## Calculate Ncut(A,B)
    #n_cut = 2.0 - Nassoc_A_B
    #
    #return n_cut

def n_cuts_recursive(affinity_mat, T1, T2):
    n = affinity_mat.shape[0]
    cluster_idx = np.zeros(n)
    current_label = 0

    def recurse(indices, label):
        if len(indices) <= T1:
            cluster_idx[indices] = label
            return

        sub_aff = affinity_mat[np.ix_(indices, indices)]
        sub_labels = n_cuts(sub_aff, 2)

        n_cut_val = calculate_n_cut_value(sub_aff, sub_labels)
        print(f'n_cut calue: {n_cut_val}')

        if n_cut_val > T2:
            cluster_idx[indices] = label
            return

        A = indices[sub_labels == 0]
        B = indices[sub_labels == 1]

        recurse(A, label)
        recurse(B, max(cluster_idx) + 1)

    recurse(np.arange(n), current_label)
    return cluster_idx

#def n_cuts_recursive(affinity_mat, T1, T2, current_label=0, max_label=[0]):
    #"""
    #Recursive normalized cuts method.
    #
    #Parameters:
    #affinity_mat: numpy array with shape [n, n], affinity/weight matrix
    #T1: int, minimum number of pixels in a cluster
    #T2: float, threshold for n_cut value
    #current_label: int, current cluster label (for internal use)
    #max_label: list with one element, maximum label used so far (for internal use)
    #
    #Returns:
    #cluster_idx: numpy array with shape [n,], cluster labels for each vertex
    #"""
    #n = affinity_mat.shape[0]
    #
    ## Base case: if cluster is too small, don't split
    #if n < T1:
        #return np.full(n, current_label, dtype=np.float64)
    #
    ## Perform binary normalized cuts (k=2)
    #binary_clusters = n_cuts(affinity_mat, k=2)
    #
    ## Calculate n_cut value for this split
    #n_cut_value = calculate_n_cut_value(affinity_mat, binary_clusters)
    #
    ## Base case: if n_cut value is below threshold, don't split
    #if n_cut_value < T2:
        #return np.full(n, current_label, dtype=np.float64)
    #
    ## Get indices for each cluster
    #unique_clusters = np.unique(binary_clusters)
    #cluster_A_indices = np.where(binary_clusters == unique_clusters[0])[0]
    #cluster_B_indices = np.where(binary_clusters == unique_clusters[1])[0]
    #
    ## Initialize result array
    #result = np.zeros(n, dtype=np.float64)
    #
    ## Recursively process cluster A
    #if len(cluster_A_indices) > 0:
        #subgraph_A = affinity_mat[np.ix_(cluster_A_indices, cluster_A_indices)]
        #max_label[0] += 1
        #label_A = max_label[0]
        #sub_result_A = n_cuts_recursive(subgraph_A, T1, T2, label_A, max_label)
        #result[cluster_A_indices] = sub_result_A
    #
    ## Recursively process cluster B
    #if len(cluster_B_indices) > 0:
        #subgraph_B = affinity_mat[np.ix_(cluster_B_indices, cluster_B_indices)]
        #max_label[0] += 1
        #label_B = max_label[0]
        #sub_result_B = n_cuts_recursive(subgraph_B, T1, T2, label_B, max_label)
        #result[cluster_B_indices] = sub_result_B
    #
    #return result
