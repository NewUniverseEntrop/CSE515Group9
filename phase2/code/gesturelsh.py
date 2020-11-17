import sys
import glob, os
import numpy as np
import math
import json
import ast
#from sets import Set
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from scipy import spatial
from gestureeddtw import *

folder = sys.argv[1]
gestureselect = sys.argv[2] # query gesture
vecoption = sys.argv[3]     # tf, tfidf
option = sys.argv[4]        # orig, pca, svd, nmf, lda
L = int(sys.argv[5]) # number of layers
k = int(sys.argv[6]) # hashes per layer

os.chdir(folder)

# load string, time series reprsentations: for edit distance and DTW
words = {}
for filename in glob.glob('*.wrd'):
    fn = os.path.splitext(filename)[0]
    with open(filename) as json_file:
        data = json.load(json_file)
        words[fn] = data

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

# start of similarity query
dist = {}
dumpfile = vecoption + option + ".pkl" # saved transformation of PCA, SVD, NMF, LDA
if option == 'orig':
    pass
elif option == 'pca':
    pca_reload = pk.load(open(dumpfile,'rb'))
    X = pca_reload.transform(X)
elif option == 'svd':
    svd_reload = pk.load(open(dumpfile,'rb'))
    X = svd_reload.transform(X)
elif option == 'nmf':
    nmf_reload = pk.load(open(dumpfile,'rb'))
    X = nmf_reload.transform(X)
elif option == 'lda':
    lda_reload = pk.load(open(dumpfile,'rb'))
    X = lda_reload.transform(X)

dim = len(X[0]) # dimension of the vectors
w = .3 # window

mu, sigma = 0, math.sqrt(k) # mean and standard deviation
p = []
b = []
for _ in range(L):
    level = []
    for _ in range(k):
        level.append(np.random.normal(mu, sigma, dim))
    p.append(level)
    b.append(np.random.uniform(0, w, k))
buckets = {}
_vectors = []

def _hash(input_point, idx):
    input_point = np.array(input_point)
    for i in range(L):
        h = []
        for j in range(k):
            h.append(math.floor((np.dot(p[i][j], input_point) + b[i][j]) / w))
        if (i, tuple(h)) not in buckets:
            buckets[(i, tuple(h))] = [idx]
        else:
            buckets[(i, tuple(h))].append(idx)

def index(vectors):
    # dim = len(vectors[0])
    for idx, vec in enumerate(vectors):
        _hash(vec, idx)

def query(input_point):
    input_point = np.array(input_point)
    candidates = set()
    overall = 0
    for i in range(L):
        h = []
        for j in range(k):
            h.append(math.floor((np.dot(p[i][j], input_point) + b[i][j]) / w))
        try:
            for candidate in buckets[(i, tuple(h))]:
                overall += 1
                candidates.add(candidate)
        except:
            pass
    return candidates, overall, len(candidates)

index(X)
[ret, overall, unique] = query(X[f2i[gestureselect]])
print('overall: ' + str(overall))
print('unique: ' + str(unique))
print([i2f[c] for c in ret])

