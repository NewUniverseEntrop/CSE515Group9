import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import math
import numpy as np
import networkx
import random
from scipy.sparse.linalg import eigs
from kmeans import performClustering
# folder = sys.argv[1]
# option = sys.argv[2]  # dot PCA SVD NMF LDA edit DTW
#
# if option == '1':
#     print()
# elif option == 2:
#     print()
# elif option == 3:
#     print()
# elif option == 4:
#     print()
# elif option == 5:
#     print()
# elif option == 6:
#     print()
# elif option == 7:
#     print()
# else:
#     print('wrong clustering option')
#
# testmatrix = np.zeros((16, 16))
# for i in range(len(testmatrix) // 4):
#     for j in range(len(testmatrix) // 4):
#         testmatrix[i, j] = 1
#         testmatrix[j, i] = 1
# for i in range(len(testmatrix) // 4, len(testmatrix) // 2):
#     for j in range(len(testmatrix) // 4, len(testmatrix) // 2):
#         testmatrix[i, j] = 1
#         testmatrix[j, i] = 1
# for i in range(len(testmatrix) // 2, len(testmatrix) // 4 * 3):
#     for j in range(len(testmatrix) // 2, len(testmatrix) // 4 * 3):
#         testmatrix[i, j] = 1
#         testmatrix[j, i] = 1
# for i in range(len(testmatrix) // 4 * 3, len(testmatrix)):
#     for j in range(len(testmatrix) // 4 * 3, len(testmatrix)):
#         testmatrix[i, j] = 1
#         testmatrix[j, i] = 1
#
#
# print(testmatrix)


def gesturecluster(matrix, k=2):
    if not matrix.shape[0] == matrix.shape[1]:
        print('matrix is not a square')
    W = np.matrix(matrix)
    D2 = np.diag([math.sqrt(1 / sum(row)) for row in matrix])
    Lsym = np.identity(W.shape[0]) - D2 * W * D2
    kvals, kvecs = eigs(Lsym, k, which = 'SM')
    V = np.mat(kvecs).real  # n*k matrix for cluster
    # kmeans = KMeans(n_clusters=k, random_state=0).fit(V)
    membershipMap = performClustering(V,4,2)
    print(membershipMap)
    # clusterresult = kmeans.labels_
    # print(clusterresult)
    return membershipMap


# gesturecluster(testmatrix, 4)  # for debugging only, no real meaning
