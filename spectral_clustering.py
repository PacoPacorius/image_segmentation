import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def spectral_clustering(affinity_mat, k):
    """
    Perform spectral clustering on a graph represented by affinity matrix.
    
    Parameters:
    affinity_mat: numpy array with shape [n, n], affinity/weight matrix
    k: int, number of clusters
    
    Returns:
    cluster_idx: numpy array with shape [n,], cluster labels for each vertex
    """
    n = affinity_mat.shape[0]
    
    # Step 1: Calculate Laplacian matrix L = D - W
    # D is diagonal matrix where D(i,i) = sum of i-th row of W
    W = affinity_mat
    degree = np.sum(W, axis=1)
    D = np.diag(degree)
    L = D - W
    
    # Step 2: Find k smallest eigenvalues and corresponding eigenvectors
    # Use scipy.sparse.linalg.eigs with which='SM' for smallest magnitude eigenvalues
    eigenvalues, eigenvectors = eigs(L, k=k, which='SM')
    
    # Sort eigenvalues and eigenvectors by eigenvalue magnitude
    idx = np.argsort(np.absolute(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 3: Create matrix U with eigenvectors as columns
    # eigenvectors is already [n, k] with eigenvectors as columns
    U = np.real(eigenvectors)  # Take real part in case of numerical issues
    
    # Step 4: Apply k-means clustering to rows of U
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    cluster_idx = kmeans.labels_
    
    return cluster_idx.astype(np.float64)
