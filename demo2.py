import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from img_to_graph import img_to_graph
from spectral_clustering import spectral_clustering

def demo2():
    """
    Demo 2: Load images, calculate affinity matrices, and perform spectral clustering
    for k=2, 3, 4 clusters on both images.
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
    k_values = [2, 3, 4]
    
    # Create figure for results
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Demo 2: Spectral Clustering on Images', fontsize=16)
    
    for img_idx, (img, img_name) in enumerate(zip(images, image_names)):
        print(f"\nProcessing {img_name}...")
        print(f"Image shape: {img.shape}")
        
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
        
        # Perform spectral clustering for different k values
        for k_idx, k in enumerate(k_values):
            print(f"  Performing spectral clustering with k={k}...")
            
            # Perform spectral clustering
            cluster_labels = spectral_clustering(affinity_mat, k)
            
            print(f"  Number of unique clusters found: {len(np.unique(cluster_labels))}")
            print(f"  Cluster distribution: {np.bincount(cluster_labels.astype(int))}")
            
            # Reshape cluster labels back to image dimensions
            M, N = img.shape[:2]
            cluster_image = cluster_labels.reshape(M, N)
            
            # Display clustering result
            im = axes[img_idx, k_idx + 1].imshow(cluster_image, cmap='tab10', vmin=0, vmax=k-1)
            axes[img_idx, k_idx + 1].set_title(f'{img_name}\nk={k} clusters')
            axes[img_idx, k_idx + 1].axis('off')
            
            # Add colorbar for the last subplot in each row
            if k_idx == len(k_values) - 1:
                plt.colorbar(im, ax=axes[img_idx, k_idx + 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemo 2 completed successfully!")

if __name__ == "__main__":
    demo2()

