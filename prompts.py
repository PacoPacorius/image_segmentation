"""

----------------

 SECTION 1 & 2

----------------

I want to implement a spectral clustering program in python. I want a function
called img_to_graph(). Its input is the image array img_array with dtype=float,
dimensions [M, N, C] and values in [0,1]. Output is the array affinty_mat,
dtype=float and dimensions [MN, MN]. The affinity matrix describes the 
image's graph. Every pixel is a vertex of the graph. Each edge is assigned a weight
with the value 1/(e^(d(i,j)), where d(i, j) is the euclidean distance between 
the i-th and the j-th pixel of the image. The graph is fully-connected and 
is undirected. The weight of an edge cannot be zero.

The above function will be used in the implementation of a spectral clustering
method. I want the function spectral_clustering(). Its inputs are the affinity matrix
affinity_mat calculated using img_to_graph() and k the number of clusters to be 
created. Its output is a one dimensional array with dtype=float and length
equal to MN named cluster_idx. It contains tags which correspond each vertex of the
input graph to a cluster. The spectral_clustering() function implements the following 
steps:
    1. Calculate the Laplacian matrix L = D - W, where W is the input affinity matrix
    and D is the diagonal matrix D(i, i) = ΣjW(i,j)
    2. Solve the eigenvalue equation Lx = λx. The eigenvalue calculation uses the
    function scipy.sparse.linalg.eigs. Only the k smallest eigenvalues are calculated.
    Find the corresponding eigenvectors of the k smallest eigenvalues.
    3. Create matrix U with dimensions [n,k] that has the eigenvectors calculated in 
    the previous step as columns. 
    4. Vector yi corresponds to the i-th row of U. Group points yi where i in [1,n]
    with the kmeans algorithm in clusters C1,...,Ck. Grouping of the points 
    uses the module sklearn as follows:
    
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters = k, random_state = 1)

        # X refers to the input data matrix
        kmeans.fit(X)

        labels = kmean.labels_


The above two functions exist each in their own .py file. I also want two
demos that showcase the program's functionality. In demo1.py, load the 
pre-calculated affinity matrix d1a from the attached file dip_hw_3.mat and 
call the function spectral_clustering() for values k=2, k=3 and k=4. Finally, 
display the results of the clustering in a graph. In demo2.py, load
the two provided images d2a and d2b from the attached file dip_hw_3.mat,
calculate the corresponding affinity matrices using img_to_graph()
and then call the spectral_clustering method for values k=2, k=3, k=4
for each of the two affinity matrices. Display the results of the 
clustering in a graph.

"""
