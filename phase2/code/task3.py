import sys
import os
import numpy as np
import json
import ast
from scipy import spatial
from lshash import lshash

folder = sys.argv[1]
gestureselect = sys.argv[2] # query gesture
L = int(sys.argv[3]) # number of layers
k = int(sys.argv[4]) # hashes per layer
t = int(sys.argv[5])
vecoption = 'tf' #sys.argv[3]     # tf, tfidf

os.chdir(folder)

# load vector representations: for PCA, SVD, NMF, LDA
if vecoption == 'tf':
    filename = folder + '/tf.txt'
elif vecoption == 'tfidf':
    filename = folder + '/tfidf.txt'
else:
    print('wrong vector model name')
with open(filename) as json_file:
    vec = json.load(json_file)

wordset = set()
gestureset = set()
# example line in vector file:   "('23', u'Y', u'11', u'[6, 6, 7]')": 0.0009615384615384616, 
for key, value in vec.items():
    li = ast.literal_eval(key)
    gestureset.add(li[0])  # document
    wordset.add((li[1], li[2], li[3])) # component + sensor + symbolic descriptor
w2i = {}
for idx, word in enumerate(wordset):
    w2i[word] = idx
gesturelist = sorted([int(v) for v in gestureset])
f2i = {} # map from document to index
i2f = {} # map from index to document
for idx, finset in enumerate(gesturelist):
    f2i[str(finset)] = idx
    i2f[idx] = str(finset)
# transform vector in dictionary to a matrix (row: word, column: file)
features = [[0.0] * len(w2i) for i in range(len(f2i))]
for key, val in vec.items():
    li = ast.literal_eval(key)
    features[f2i[li[0]]][w2i[(li[1], li[2], li[3])]] = val
X = np.array(features)

# initialize
lsh = lshash(L, k)
# build index
lsh.index(X)
# retrieval
q_vec = X[f2i[gestureselect]]
[ret, overall, unique] = lsh.query(q_vec)

# sort the results according to Euclidean distance
dist = {}
for idx in ret:
    dist[i2f[idx]] = spatial.distance.euclidean(q_vec, X[idx])
rank = [k for k, v in sorted(dist.items(), key = lambda item : item[1])]
rank = [rank[i] for i in range(min(t, len(rank)))]
print('overall: ' + str(overall))
print('unique: ' + str(unique))
#print(",".join(rank))
rank = [int(r) for r in rank]
print(rank)