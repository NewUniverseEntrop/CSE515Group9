import sys
import glob, os
import numpy as np
import math
import json
import csv
import ast
from collections import Counter
#from sets import Set
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from scipy import spatial
from gestureeddtw import *
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd
from kmeans import runKmeanClustering
from gestureclusterlaplacian import gesturecluster
folder = sys.argv[1]
vecoption = sys.argv[2] # tf, tfidf
option = sys.argv[3]    # dotp, pca, svd, nmf, lda, ed, dtw
topp = int(sys.argv[4])
grouping_strategy = sys.argv[5]

os.chdir(folder)

# load string, time series reprsentations
words = {}
for filename in glob.glob('*.wrd'):
    fn = os.path.splitext(filename)[0]
    with open(filename) as json_file:
        data = json.load(json_file)
        words[fn] = data

# load vector representations
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

for key, value in vec.items():
    li = ast.literal_eval(key)
    gestureset.add(li[0])
    wordset.add((li[1], li[2], li[3]))

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
print(len(features), len(features[0]))
X = np.array(features)

distmatrix = [[0.0] * len(f2i) for _ in range(len(f2i))]
print(distmatrix)

dumpfile = vecoption + option + ".pkl"
if option == 'dotp':
    for i in range(len(f2i)):
        fea1 = features[i]
        for j in range(i, len(f2i)):
            fea2 = features[j]
            distmatrix[i][j] = distmatrix[j][i] = np.dot(fea1, fea2)
elif option == 'pca':
    pca_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = pca_reload .transform(X)
    for i in range(len(f2i)):
        fea1 = X_reduced[i]
        for j in range(i, len(f2i)):
            fea2 = X_reduced[j]
            distmatrix[i][j] = distmatrix[j][i] = 1 - spatial.distance.cosine(fea1, fea2)
elif option == 'svd':
    svd_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = svd_reload.transform(X)
    for i in range(len(f2i)):
        fea1 = X_reduced[i]
        for j in range(i, len(f2i)):
            fea2 = X_reduced[j]
            distmatrix[i][j] = distmatrix[j][i] = 1 - spatial.distance.cosine(fea1, fea2)
elif option == 'nmf':
    nmf_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = nmf_reload.transform(X)
    for i in range(len(f2i)):
        fea1 = X_reduced[i]
        for j in range(i, len(f2i)):
            fea2 = X_reduced[j]
            distmatrix[i][j] = distmatrix[j][i] = 1 - spatial.distance.cosine(fea1, fea2)
elif option == 'lda':
    lda_reload = pk.load(open(dumpfile,'rb'))
    X_reduced = lda_reload.transform(X)
    for i in range(len(f2i)):
        fea1 = X_reduced[i]
        for j in range(i, len(f2i)):
            fea2 = X_reduced[j]
            distmatrix[i][j] = distmatrix[j][i] = 1 - spatial.distance.cosine(fea1, fea2)
elif option == 'ed':
    datakey = 'winsymb'
    for i in range(len(f2i)):
        gesture1 = words[i2f[i]]
        for j in range(i, len(f2i)):
            gesture2 = words[i2f[j]]
            print(i, j)
            series1 = []
            series2 = []
            avg1, avg2 = [], []
            std1, std2 = [], []
            for component, data in gesture2.items():
                for sensor, wins in data.items():
                    series2.append([ast.literal_eval(v) for k, v in sorted(wins[datakey].items(), key=lambda item: int(item[0]))])
                    series1.append([ast.literal_eval(v) for k, v in sorted(gesture1[component][sensor][datakey].items(), key=lambda item: int(item[0]))])
                    avg1.append(gesture1[component][sensor]['avg'])
                    avg2.append(wins['avg'])
                    std1.append(gesture1[component][sensor]['std'])
                    std2.append(wins['std'])
            distmatrix[i][j] = distmatrix[j][i] = editdist(series1, series2, avg1, avg2, std1, std2)
elif option == 'dtw':
    datakey = 'winavg'
    for i in range(len(f2i)):
        gesture1 = words[i2f[i]]
        for j in range(i, len(f2i)):
            gesture2 = words[i2f[j]]
            series1 = []
            series2 = []
            avg1, avg2 = [], []
            std1, std2 = [], []
            for component, data in gesture2.items():
                for sensor, wins in data.items():
                    series2.append([v for k, v in sorted(wins[datakey].items(), key=lambda item: int(item[0]))])
                    series1.append([v for k, v in sorted(gesture1[component][sensor][datakey].items(), key=lambda item: int(item[0]))])
                    avg1.append(gesture1[component][sensor]['avg'])
                    avg2.append(wins['avg'])
                    std1.append(gesture1[component][sensor]['std'])
                    std2.append(wins['std'])
            distmatrix[i][j] = distmatrix[j][i] = dtw(series1, series2, avg1, avg2, std1, std2)

# convert distance to similarity
if option == 'ed' or option == 'dtw':
    mx, mn = max(max(distmatrix)), min(min(distmatrix))
    scale = mx - mn
    distmatrix = [[1 - (ele - mn) / scale for ele in row] for row in distmatrix]
print("dist",np.array(distmatrix).shape)

def getWordScoreMatrixForLatentFeature(word_score_df,i2w):
    column_names = []
    for index,word in  i2w.items():
        column_names.append(str(word))
    df = pd.DataFrame([word_score_df.T.to_numpy()], columns=column_names)
    output_df = df.T.sort_values(by=0,ascending=False)
    return output_df.T

def getLatentFeaturesAsWordScore(components,latent_features_file,topk):
    print(components.shape)
    word_score_df = pd.DataFrame(components)

    for i in range(topk):
        latent_df = getWordScoreMatrixForLatentFeature(word_score_df.iloc[i], i2f)
        latent_df.to_csv(latent_features_file + "." + str(i) + ".csv")
        # print(latent_df)

def svd(distmatrix):
    # decomposition using SVD
    latent_features_file = folder + '/' + vecoption + "." + option + "." + grouping_strategy
    print("sim matrix",len(distmatrix), len(distmatrix[0]))
    u,s,v = np.linalg.svd(distmatrix)
    print("SVD")
    print(v[:topp])
    getLatentFeaturesAsWordScore(v[:topp], latent_features_file, topp)  #task3a
    # print("u",len(u), len(u[0]))
    # print("top",u[:, 0 : topp])
    # print("s",s)
    print([np.argmax(a) for a in u[ :, 0 : topp]])
    membership = {}  #task 4a
    for i in range(topp):
        membership[i] = []
    for index,a in enumerate(u[:, 0: topp]):
        membership[np.argmax(a)].append(i2f[index])
    print(membership)


def nmf(distmatrix):
    # decomposition using NMF
    latent_features_file = folder + '/' + vecoption + "." + option + "." + grouping_strategy
    model = NMF(n_components=topp, init='random', random_state=0)
    W = model.fit_transform(distmatrix)
    H = model.components_
    print(W)
    getLatentFeaturesAsWordScore(H, latent_features_file, topp)  #task3b
    print([np.argmax(a) for a in W])
    membership = {}  #task 4b
    for i in range(topp):
        membership[i] = []
    for index, a in enumerate(W):
        membership[np.argmax(a)].append(i2f[index])
    print(membership)

def kmeans(distmatrix):  #task 4c
    membership_indices_map = runKmeanClustering(np.array(distmatrix), topp, 2)
    membership = {}
    for i in range(topp):
        membership[i] = []
    for key, value in membership_indices_map.items():
        for i in value:
            membership[key].append(i2f[i])
    print(membership)
def specteral_clustering(distmatrix):
    labels = gesturecluster(np.array(distmatrix), topp)
    membership = {}
    for i in range(topp):
        membership[i] = []
    for i,label in enumerate(labels):
        membership[label].append(i2f[i])
    print(membership)

if grouping_strategy == 'svd':
    svd(distmatrix)
elif grouping_strategy == 'nmf':
    nmf(distmatrix)
elif grouping_strategy == 'kmeans':
    kmeans(distmatrix)
elif grouping_strategy == 'spectral':
    specteral_clustering(distmatrix)