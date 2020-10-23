from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import sys
import numpy as np
import pandas as pd
import json
import ast
#from sets import Set
import pickle as pk

folder = sys.argv[1]
vecoption = sys.argv[2]  # tf, tf-idf
topk = int(sys.argv[3])  # user designated number of latent features
option = sys.argv[4]     # pca, svd, nfm, lda

if vecoption == 'tf':
    filename = folder + '/tf.txt'
elif vecoption == 'tfidf':
    filename = folder + '/tfidf.txt'
else:
    print('wrong vector model name')
with open(filename) as json_file:
    vec = json.load(json_file)

# load vector representation
wordset = set()
gestureset = set()

# example line in vector file:   "('23', u'Y', u'11', u'[6, 6, 7]')": 0.0009615384615384616, 
for key, value in vec.items():
    li = ast.literal_eval(key)
    gestureset.add(li[0]) # document
    wordset.add((li[1], li[2], li[3]))  # component + sensor + symbolic descriptor

w2i = {} # map from word to index
i2w = {}  # map from index to word
for idx, word in enumerate(wordset):
    w2i[word] = idx
    i2w[idx] = word

gesturelist = sorted([int(v) for v in gestureset])
f2i = {} # map from document to index
i2f = {} # map from index to document
for idx, finset in enumerate(gesturelist):
    f2i[str(finset)] = idx
    i2f[idx] = str(finset)

# transform vector in dictionary to a matrix (row: word, column: document)
features = [[0.0] * len(w2i) for i in range(len(f2i))]
for key, val in vec.items():
    li = ast.literal_eval(key)
    features[f2i[li[0]]][w2i[(li[1], li[2], li[3])]] = val

print(features)
X = np.array(features)
print(X)


def getWordScoreMatrixForLatentFeature(word_score_df,i2w):
    column_names = []
    for index,word in  i2w.items():
        column_names.append(str(word))
    df = pd.DataFrame([word_score_df.T.to_numpy()], columns=column_names)
    output_df = df.T.sort_values(by=0,ascending=False)
    return output_df.T

def getLatentFeaturesAsWordScore(components,latent_features_file):
    print(components.shape)
    word_score_df = pd.DataFrame(components)

    for i in range(topk):
        latent_df = getWordScoreMatrixForLatentFeature(word_score_df.iloc[i], i2w)
        latent_df.to_csv(latent_features_file + "." + str(i) + ".csv")
        print(latent_df)



dumppath = folder + '/' + vecoption + option + ".pkl"

latent_features_file = folder + '/' + vecoption + "." + option
if option == 'pca':
    pca = PCA(n_components = topk)
    pca.fit(X)
    getLatentFeaturesAsWordScore(pca.components_,latent_features_file)
    pk.dump(pca, open(dumppath,"wb"))
elif option == 'svd':
    svd = TruncatedSVD(n_components=topk)
    svd.fit(X)
    getLatentFeaturesAsWordScore(svd.components_,latent_features_file)
    pk.dump(svd, open(dumppath,"wb"))
elif option == 'nmf':
    nmf = NMF(n_components=topk)
    nmf.fit(X)
    getLatentFeaturesAsWordScore(nmf.components_,latent_features_file)
    pk.dump(nmf, open(dumppath,"wb"))
elif option == 'lda':
    lda = LatentDirichletAllocation(n_components=topk,max_iter=10)
    lda.fit(X)
    print(lda.components_)
    getLatentFeaturesAsWordScore(lda.components_,latent_features_file)
    pk.dump(lda, open(dumppath,"wb"))
else:
    print('wrong decomposition option')