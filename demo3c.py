import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from img_to_graph import img_to_graph
from n_cuts import n_cuts_recursive, n_cuts, calculate_n_cut_value

def demo3c():
    """
    Demo 3c: Complete recursive normalized cuts on images d2a and d2b.
    First manually splits image into 2 clusters using n_cuts(), then applies
    n_cuts_recursive() to each resulting cluster. Uses T1=5 and T2=0.20 as thresholds.
    """
    
    # Load the images from .mat file
    try:
        data = loadmat('dip_hw_3.mat')
        img1 = data['d2a']
        img2 = data['d2b']
        print(f"Loaded image 1 with shape: {img1.shape}")
        print(f"Loaded image 2 with shape: {img2.shape}")
    except FileNotFoundError:
        print("Error: dip_hw_3.mat file not found!")
        print("Please make sure the file is in the same directory as this script.")
        return
    except KeyError:
        print("Error: 'd2a' or 'd2b' keys not found in the .mat file!")
        return
    
    # Ensure images are in correct format [M, N, C] with values in [0,1]
    def preprocess_image(img):
        # Convert to float and normalize to [0,1] if needed
        img = img.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Ensure 3D array [M, N, C]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)  # Add channel dimension
        
        return img
    
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    
    images = [img1, img2]
    image_names = ['Image 1 (d2a)', 'Image 2 (d2b)']
    
    # Set thresholds
    T1 = 5   # Minimum number of pixels in a cluster
    T2 = 0.20  # Threshold for n_cut value
    
    # Create figure for results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Demo 3c: Complete Recursive Normalized Cuts (T1={T1}, T2={T2})', fontsize=16)
    
    for img_idx, (img, img_name) in enumerate(zip(images, image_names)):
        print(f"\nProcessing {img_name}...")
        print(f"Image shape: {img.shape}")
        print(f"Using thresholds: T1={T1}, T2={T2}")
        
        # Display original image
        if img.shape[2] == 1:
            axes[img_idx, 0].imshow(img[:,:,0], cmap='gray')
        else:
            axes[img_idx, 0].imshow(img)
        axes[img_idx, 0].set_title(f'Original {img_name}')
        axes[img_idx, 0].axis('off')
        
        # Calculate affinity matrix
        print("Calculating affinity matrix...")
        affinity_mat = img_to_graph(img)
        print(f"Affinity matrix shape: {affinity_mat.shape}")
        
        # Step 1: Manually split the image into 2 clusters using n_cuts()
        print("Step 1: Manual binary split using n_cuts()...")
        initial_clusters = n_cuts(affinity_mat, k=2)
        print('ncut_value: ', calculate_n_cut_value(affinity_mat, initial_clusters))
        
        # Get indices for each initial cluster
        unique_initial = np.unique(initial_clusters)
        cluster_A_indices = np.where(initial_clusters == unique_initial[0])[0]
        cluster_B_indices = np.where(initial_clusters == unique_initial[1])[0]
        
        print(f"Initial split - Cluster A: {len(cluster_A_indices)} pixels, Cluster B: {len(cluster_B_indices)} pixels")
        
        # Initialize final result array
        final_cluster_labels = np.zeros(len(initial_clusters), dtype=np.float64)
        max_label = 0
        
        # Step 2: Apply n_cuts_recursive() to each resulting cluster
        print("Step 2: Applying recursive n_cuts to each cluster...")
        
        # Process Cluster A
        if len(cluster_A_indices) > 0:
            print(f"  Processing Cluster A ({len(cluster_A_indices)} pixels)...")
            subgraph_A = affinity_mat[np.ix_(cluster_A_indices, cluster_A_indices)]
            sub_result_A = n_cuts_recursive(subgraph_A, T1, T2)
            # Adjust labels to be unique globally
            sub_result_A = sub_result_A + max_label
            final_cluster_labels[cluster_A_indices] = sub_result_A
            max_label = int(np.max(sub_result_A)) + 1
            print(f"    Cluster A resulted in {len(np.unique(sub_result_A))} sub-clusters")
        
        # Process Cluster B
        if len(cluster_B_indices) > 0:
            print(f"  Processing Cluster B ({len(cluster_B_indices)} pixels)...")
            subgraph_B = affinity_mat[np.ix_(cluster_B_indices, cluster_B_indices)]
            sub_result_B = n_cuts_recursive(subgraph_B, T1, T2)
            # Adjust labels to be unique globally
            sub_result_B = sub_result_B + max_label
            final_cluster_labels[cluster_B_indices] = sub_result_B
            print(f"    Cluster B resulted in {len(np.unique(sub_result_B))} sub-clusters")
        
        # Get final statistics
        unique_clusters = np.unique(final_cluster_labels)
        cluster_counts = np.bincount(final_cluster_labels.astype(int))
        
        print(f"Final result: {len(unique_clusters)} total clusters")
        print(f"Cluster distribution: {cluster_counts[cluster_counts > 0]}")
        print(f"Cluster labels range: {final_cluster_labels.min():.0f} to {final_cluster_labels.max():.0f}")
        
        # Reshape cluster labels back to image dimensions
        M, N = img.shape[:2]
        cluster_image = final_cluster_labels.reshape(M, N)
        
        # Display clustering result
        num_clusters = len(unique_clusters)
        im = axes[img_idx, 1].imshow(cluster_image, cmap='tab20', vmin=0, vmax=max(num_clusters-1, 1))
        axes[img_idx, 1].set_title(f'{img_name}\nRecursive N-cuts\n{num_clusters} clusters')
        axes[img_idx, 1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[img_idx, 1], fraction=0.046, pad=0.04)
        
        # Print some statistics about the clustering
        print(f"Cluster sizes: {sorted(cluster_counts[cluster_counts > 0], reverse=True)}")
        
        # Check if any clusters are smaller than T1
        small_clusters = np.sum(cluster_counts < T1) - np.sum(cluster_counts == 0)
        if small_clusters > 0:
            print(f"Note: {small_clusters} clusters have fewer than T1={T1} pixels (stopped due to T2 threshold)")
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemo 3c completed successfully!")

if __name__ == "__main__":
    demo3c()
