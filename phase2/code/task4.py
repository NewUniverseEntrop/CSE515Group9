import sys
import glob, os
import numpy as np
import json
import ast
import pickle as pk
from scipy import spatial
from lshash import lshash

folder = sys.argv[1]
gestureselect = sys.argv[2] # query gesture
vecoption = sys.argv[3]     # tf, tfidf
relevant = list(sys.argv[4].split(','))
irrelevant = list(sys.argv[5].split(','))

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

relevant = [f2i[r] for r in relevant]
irrelevant = [f2i[ir] for ir in irrelevant]

# Salton and Buckley
# Qnew = log[pi(1-ui)/ui(1-pi)]
# pi = P(xi|rel) = (ri+0.5)/(R+1.0)
# ui = P(xi|nonrel) = (nri+0.5)/(NR+1.0)

# normalize the features to [0, 1], suitable for the S&B algorithm
features_norm = np.array(features)
features_norm = features_norm / features_norm.max(axis=0)
# average feature values in relevant objects
pi = (np.sum(features_norm[relevant, :], axis = 0) + 0.5) / (len(relevant) + 1)
# average feature values in irrelevant objects
ui = (np.sum(features_norm[irrelevant, :], axis = 0) + 0.5) / (len(irrelevant) + 1)
# adjust the query and transform back to original scale
Qnew = np.log(pi * (1 - ui) / (ui * (1 - pi))) * np.max(features, axis = 0)

q = features[f2i[gestureselect]]

dist1 = {}
dist2 = {}
dist3 = {}
for gesture2, fea2 in enumerate(features):
    dist1[i2f[gesture2]] = spatial.distance.euclidean(q, fea2)
    dist2[i2f[gesture2]] = spatial.distance.euclidean(Qnew, fea2)
    dist3[i2f[gesture2]] = spatial.distance.euclidean(q + Qnew, fea2)
dist1 = [(k, v) for k, v in sorted(dist1.items(), key = lambda item : item[1])]
dist2 = [(k, v) for k, v in sorted(dist2.items(), key = lambda item : item[1])]
dist3 = [(k, v) for k, v in sorted(dist3.items(), key = lambda item : item[1])]
print('original')
print(dist1[0: 10])
print('revised')
print(dist2[0: 10])
print('combined')
print(dist3[0: 10])