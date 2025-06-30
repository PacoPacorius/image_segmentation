import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from spectral_clustering import spectral_clustering

def demo1():
    """
    Demo 1: Load pre-calculated affinity matrix and perform spectral clustering
    for k=2, 3, 4 clusters.
    """
    
    # Load the pre-calculated affinity matrix from .mat file
    try:
        data = loadmat('dip_hw_3.mat')
        affinity_matrix = data['d1a']
        print(f"Loaded affinity matrix with shape: {affinity_matrix.shape}")
    except FileNotFoundError:
        print("Error: dip_hw_3.mat file not found!")
        print("Please make sure the file is in the same directory as this script.")
        return
    except KeyError:
        print("Error: 'd1a' key not found in the .mat file!")
        return
    
    # Test spectral clustering for k=2, 3, 4
    k_values = [2, 3, 4]
    
    # Create subplots for visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Demo 1: Spectral Clustering Results', fontsize=16)
    
    for i, k in enumerate(k_values):
        print(f"\nPerforming spectral clustering with k={k}...")
        
        # Perform spectral clustering
        cluster_labels = spectral_clustering(affinity_matrix, k)
        
        print(f"Number of unique clusters found: {len(np.unique(cluster_labels))}")
        print(f"Cluster distribution: {np.bincount(cluster_labels.astype(int))}")
        
        # Visualize results
        # Since we don't know the original image dimensions, we'll create a simple visualization
        n = len(cluster_labels)
        
        # Display as 1D plot 
        axes[i].plot(cluster_labels, 'o-', markersize=2)
        axes[i].set_title(f'k={k} (1D representation)')
        axes[i].set_xlabel('Vertex index')
        axes[i].set_ylabel('Cluster label')
        
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemo 1 completed successfully!")

if __name__ == "__main__":
    demo1()
