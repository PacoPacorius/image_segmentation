import numpy as np
from scipy.spatial.distance import cdist

def img_to_graph(img_array):
    """
    Convert an image to a graph representation using affinity matrix.
    
    Parameters:
    img_array: numpy array with shape [M, N, C], dtype=float, values in [0,1]
    
    Returns:
    affinity_mat: numpy array with shape [MN, MN], dtype=float
                 Affinity matrix where each entry (i,j) = 1/exp(d(i,j))
                 d(i,j) is euclidean distance between pixel i and pixel j
    """
    M, N, C = img_array.shape
    
    # Reshape image to [MN, C] - each row is a pixel with C channels
    pixels = img_array.reshape(M * N, C)
    
    # Calculate euclidean distances between all pairs of pixels
    # cdist returns a [MN, MN] matrix where entry (i,j) is distance between pixel i and j
    distances = cdist(pixels, pixels, metric='euclidean')
    
    # Convert distances to affinity weights: weight = 1/exp(distance)
    affinity_mat = 1.0 / (np.exp(distances))
    
    # Ensure diagonal elements are set to maximum affinity (distance = 0 for same pixel)
    np.fill_diagonal(affinity_mat, 1.0)
    
    return affinity_mat.astype(np.float64)
