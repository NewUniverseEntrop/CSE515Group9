import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
import networkx
import random
from scipy.sparse.linalg import eigs

folder = sys.argv[1]
option = sys.argv[2]  # dot PCA SVD NMF LDA edit DTW

if option == '1':
    print()
elif option == 2:
    print()
elif option == 3:
    print()
elif option == 4:
    print()
elif option == 5:
    print()
elif option == 6:
    print()
elif option == 7:
    print()
else:
    print('wrong clustering option')

testmatrix = np.zeros((10, 10))
for i in range(len(testmatrix) // 2):
    for j in range(len(testmatrix) // 2):
        testmatrix[i, j] = 1
        testmatrix[j, i] = 1
for i in range(len(testmatrix) // 2, len(testmatrix)):
    for j in range(len(testmatrix) // 2, len(testmatrix)):
        testmatrix[i, j] = 1
        testmatrix[j, i] = 1


#print(testmatrix)


def gesturecluster(matrix, k=2):
    if not matrix.shape[0] == matrix.shape[1]:
        print('matrix is not a square')
    G = networkx.Graph()
    for i in range(len(matrix)):
        G.add_node(str(i))
    for i in range(len(matrix)):
        #subvector = matrix[i, :]
        #knearest = np.argpartition(subvector, -knear)[-knear - 1:-1]  # k nearest
        for j in range(len(matrix)):
            if matrix[i][j] > 0:  # graph weight need to be positive
                G.add_edge(str(i), str(j), weight=matrix[i][j])  # build k-nearest graph
    normalaplacian = networkx.normalized_laplacian_matrix(G)
    kvals, kvecs = eigs(normalaplacian, k, which = 'SM')
    V = abs(np.mat(kvecs))  # k*n matrix for cluster
    kmeans = KMeans(n_clusters=k, random_state=0).fit(V)
    clusterresult = kmeans.labels_
    #print(clusterresult)
    return clusterresult


gesturecluster(testmatrix, 2)  # for debugging only, no real meaning
