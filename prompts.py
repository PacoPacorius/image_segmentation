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


------------

 SECTION 3

------------

I also want to implement the normalized cuts method in python. I want three functions,
n_cuts(), calculate_n_cut_value() and n_cuts_recursive(). I'll explain each function
in order.

n_cuts() implements the non-recursive normalized cuts method. Its inputs are an
affinity matrix that describes the graph of an image and the number of clusters to 
be created k. Its output is the one-dimensional array with dtype=float and length 
equal to MN named cluster_idx. It contains tags which correspond each vertex of the 
input graph to a cluster. The n_cuts() method implements the following steps:

    1. Calculate the Laplacian matrix L = D - W, where W is the input affinity matrix
    and D is the diagonal matrix D(i, i) = ΣjW(i,j)
    2. Solve the generalized eigenvalue equation Lx = λDx. The eigenvalue calculation uses the
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

n_cuts_recursive() implements the recursive normalized cuts method for k=2. 
Its inputs are an affinity matrix, integer threshold T1 which is the minimum 
number of pixels in a cluster and floating point threshold T2 which is the 
threshold for the n_cut value (more on how to calculate this later). The 
function's output is the one-dimensional array with dtype=float and length 
equal to MN named cluster_idx. It contains tags which correspond each vertex of the 
input graph to a cluster. The functions implements the steps described in the
n_cuts() function, but every time the steps are completed, the image's 
graph is split in two and the n_cuts_recursive() function is called 
recursively for each of the two graphs. The splitting is like the creation of a
(possibly unbalanced) binary tree where each vertex holds the tag information 
in relation to its parent. The recursion ends if both clusters
satisfy the following conditions: 

    a. if the points of the cluster are less than the threshold T1 or 
    b. if the value n_cut is smaller than the threshold T2.

The function calculate_n_cut_value() calculates the n_cut value mentioned in
the n_cuts_recursive() function. Its inputs are an affinity matrix and a 
cluster_idx array. Its output is floating point value n_cut. This function
implements the following calculation for two clusters A and B:

    Ncut(A,B) = 2 - Nassoc(A,B)

    where Nassoc(A,B) = assoc(A,A)/assoc(A,V) + assoc(B,B)/assoc(B,V)

    where assoc(A,V) = Σu(ΣtW(u,t)), u iterates over all weights of edges in cluster A
    and t iterates over all weights of edges in the image's graph. W is the affinity
    matrix. Finally set the output value n_cut = Ncut(A,B).

All of the above functions are in the file n_cuts.py. In addition to these functions
I want three demos, each in their own file. In the demo3a.py file the n_cuts() function
is called for values k=2,3,4 for both images d2a and d2b supplied by the file dip_hw_3.mat.
It then displays the the results of the clustering in a graph in relation to the 
original images. In demo3b.py the recursive normalized cuts method is used but 
only for one recursive step for each of the images d2a and d2b. In other words, 
the graph that describes each image is split in two. Display the n_cut value 
and the results of the clustering in a graph compared to the original images.
In demo3c.py the completed recursive normalized cuts method is executed for both 
of the images d2a and d2b. Set T1=5 and T2=0.20. Display the results of the 
clustering in a graph.

"""
