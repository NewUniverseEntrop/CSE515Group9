import sys
import glob, os
import numpy as np
import math
import json
import ast
from sets import Set
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
option = sys.argv[4]        # dotp, pca, svd, nmf, lda, ed, dtw

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

wordset = Set()
gestureset = Set()
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
if option == 'dotp':
    fea1 = features[f2i[gestureselect]]
    for gesture2, fea2 in enumerate(features):
        dist[i2f[gesture2]] = np.dot(fea1, fea2)
    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : -item[1])]
elif option == 'pca':
    pca_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = pca_reload .transform(X)
    fea1 = X_reduced[f2i[gestureselect]]
    for gesture2, fea2 in enumerate(X_reduced):
        dist[i2f[gesture2]] = spatial.distance.euclidean(fea1, fea2)
    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : item[1])]
elif option == 'svd':
    svd_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = svd_reload.transform(X)
    fea1 = X_reduced[f2i[gestureselect]]
    for gesture2, fea2 in enumerate(X_reduced):
        dist[i2f[gesture2]] = spatial.distance.euclidean(fea1, fea2)
    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : item[1])]
elif option == 'nmf':
    nmf_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = nmf_reload.transform(X)
    fea1 = X_reduced[f2i[gestureselect]]
    for gesture2, fea2 in enumerate(X_reduced):
        dist[i2f[gesture2]] = spatial.distance.euclidean(fea1, fea2)
    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : item[1])]
elif option == 'lda':
    lda_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = lda_reload.transform(X)
    fea1 = X_reduced[f2i[gestureselect]]
    for gesture2, fea2 in enumerate(X_reduced):
        dist[i2f[gesture2]] = spatial.distance.euclidean(fea1, fea2)
    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : item[1])]
elif option == 'ed':
    datakey = 'winsymb'
    gesture1 = words[gestureselect]
    for gesture, value in words.items():
        series1 = []
        series2 = []
        avg1, avg2 = [], []
        std1, std2 = [], []
        for component, data in value.items():
            for sensor, wins in data.items():
                # ordering of the windows was not preserved in the file, so we sort them here
                series2.append([ast.literal_eval(v) for k, v in sorted(wins[datakey].items(), key=lambda item: int(item[0]))])
                series1.append([ast.literal_eval(v) for k, v in sorted(gesture1[component][sensor][datakey].items(), key=lambda item: int(item[0]))])
                avg1.append(gesture1[component][sensor]['avg'])
                avg2.append(wins['avg'])
                std1.append(gesture1[component][sensor]['std'])
                std2.append(wins['std'])
        dist[gesture] = editdist(series1, series2, avg1, avg2, std1, std2)

    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : item[1])]
elif option == 'dtw':
    datakey = 'winavg'
    gesture1 = words[gestureselect]
    for gesture, value in words.items():
        series1 = []
        series2 = []
        for component, data in value.items():
            for sensor, wins in data.items():
                # ordering of the windows was not preserved in the file, so we sort them here
                series2.append([v for k, v in sorted(wins[datakey].items(), key=lambda item: int(item[0]))])
                series1.append([v for k, v in sorted(gesture1[component][sensor][datakey].items(), key=lambda item: int(item[0]))])
        dist[gesture] = dtw(series1, series2)

    dist = [(k, v) for k, v in sorted(dist.items(), key = lambda item : item[1])]

print(dist[0 : 10])

