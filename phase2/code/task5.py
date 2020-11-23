# run phase 2 task3 to generate similarity matrix before running PPR
#   gesturedisc.py /Users/xavi/MWDB/xin/CSE515Group9/phase2/3_class_gesture_data tfidf dtw 3 svd
# task1.py /Users/xavi/MWDB/xin/CSE515Group9/phase2/3_class_gesture_data 2 10 1,2,3,4,5
# task1.py cmd parameters :  dir_path k m and user selected gestures


import json
import numpy as np
import sys,os
folder = sys.argv[1]
os.chdir(folder)
relevant = [] if sys.argv[2] == "-1" else list(sys.argv[2].split(','))
irrelevant = [] if sys.argv[3] == "-1" else list(sys.argv[3].split(','))
neutral = [] if sys.argv[4] == "-1" else list(sys.argv[4].split(','))
try:
    k = int(sys.argv[5])
except:
    k = 4

def getNearestNeightbours(arr,k=2):
    #print(arr)
    if np.sum(arr) == 0:
        return arr
    indices = np.argsort(arr)[-k:]
    for i,x in enumerate(arr):
        if i not in indices:
            arr[i] = 0
    return arr/np.sum(arr)


def PPR(adjacency_matrix,seed_vector,max_iterations=100,beta_probabilty=0.15):
    n = adjacency_matrix.shape[0]
    seed_vector = seed_vector/np.sum(seed_vector)
    x = np.full((n,1),1/n)
    #print(x)
    #print(seed_vector.shape)
    #print("seed")
    
    for i in range(max_iterations):
        x_prev = x
        a = (1-beta_probabilty)*np.matmul(adjacency_matrix,x)
        b = seed_vector*beta_probabilty
        x = np.add(a,b)
        #print(x)
        sum = np.sum(np.abs(x-x_prev))
        if(sum < 1.0e-6):
            #print(i,sum)
            break
    return x

similarity_matrix = np.load('similarity_matrix_dtw.npy')
for i in range(len(similarity_matrix)):
    similarity_matrix[i][i] = 0
# print(similarity_matrix)

# k = 4
seed_vector = np.ones(similarity_matrix.shape[0]).reshape(-1,1) * 0.1
with open('f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open('i2f.dump', 'r') as fp:
    i2f = json.load(fp)

relevant = [f2i[r] for r in relevant]
irrelevant = [f2i[r] for r in irrelevant]
neutral = [f2i[r] for r in neutral]

full = relevant + irrelevant + neutral

for gesture in relevant:
    seed_vector[gesture] = 1
for gesture in irrelevant:
    seed_vector[gesture] = 0

transformed_similarity_matrix= np.apply_along_axis(getNearestNeightbours,axis=0,arr=similarity_matrix,k=k)
#print(transformed_similarity_matrix)
ranks = PPR(transformed_similarity_matrix,seed_vector,max_iterations=100)
#print(ranks)
ranks = ranks.reshape(-1)

ranks = ranks.argsort()[::-1]
res = []
for r in ranks:
    if r in full:
        res.append(i2f[str(r)])
print(res)