import sys
import glob, os
import numpy as np
import json
import ast
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from scipy import spatial
from lshash import lshash

folder = sys.argv[1]
gestureselect = sys.argv[2] # query gesture
vecoption = sys.argv[3]     # tf, tfidf
option = sys.argv[4]        # orig, pca, svd, nmf, lda
L = int(sys.argv[5]) # number of layers
k = int(sys.argv[6]) # hashes per layer
t = int(sys.argv[7])

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

lsh = lshash(L, k)
lsh.index(X)
q_vec = X[f2i[gestureselect]]
[ret, overall, unique] = lsh.query(q_vec)

dist = {}
for idx in ret:
    dist[i2f[idx]] = spatial.distance.euclidean(q_vec, X[idx])
rank = [k for k, v in sorted(dist.items(), key = lambda item : item[1])]
rank = [rank[i] for i in range(min(t, len(rank)))]
print('overall: ' + str(overall))
print('unique: ' + str(unique))
print(rank)
