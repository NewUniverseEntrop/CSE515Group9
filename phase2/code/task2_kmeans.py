import pandas as pd
import numpy as np
import pickle
from math import sqrt
import sys
import json
import ast
import functools


def custom_sort(word1, word2):
    word1_1 = word1.split("_")
    word2_2 = word2.split("_")
    if len(word1_1) == 1 and len(word2_2) == 1:
        return int(word1_1[0]) - int(word2_2[0])
    elif len(word1_1) > 1 and len(word2_2) > 1:
        if word1_1[0] == word2_2[0]:
            return int(word1_1[1]) - int(word2_2[1])
        else:
            return int(word1_1[0]) - int(word2_2[0])
    elif len(word1_1) > 1 and len(word2_2) == 1:
        if word1_1[0] == word2_2[0]:
            return 1
        else:
            return int(word1_1[0]) - int(word2_2[0])
    else:
        if word1_1[0] == word2_2[0]:
            return -1
        else:
            return int(word1_1[0]) - int(word2_2[0])


def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for index,train_row in train.iterrows():
        dist = euclidean_distance(test_row, train_row)
        distances.append((i2f[str(index)], dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    neighbors = distances[:num_neighbors]
    return [row[0] for row in neighbors]


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1.iloc[i] - row2.iloc[i])**2
    return sqrt(distance)


def get_actual_result(train_labels, index):
    # print("")
    index_split = index.split("_")[0]
    class_ = train_labels.loc[train_labels[train_labels.columns[0]] == index_split]
    class_value = class_.iloc[0].iloc[1]
    return class_value
    

# folder = sys.argv[1]
folder = "C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\3_class_gesture_data\\"
# vecoption = sys.argv[2]  # tf, tf-idf
vecoption = 'tfidf'
# topk = int(sys.argv[3])  # user designated number of latent features
topk = 4
# option = sys.argv[4]     # pca, svd, nfm, lda
option = 'pca'

with open('C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\code\\f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open('C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\code\\i2f.dump', 'r') as fp:
    i2f = json.load(fp)


train_labels = pd.read_excel(folder+'/labels.xlsx', header=None).astype(str)
dataset = pd.read_csv(folder+'/train_'+option+'_'+vecoption+'.csv', header=None)
train_dataset = pd.DataFrame()

for index, row in train_labels.iterrows():
    index_in_dataset = f2i[str(row.iloc[0])]
    train_dataset = train_dataset.append(dataset.iloc[index_in_dataset])

test_dataset = pd.concat([dataset,train_dataset]).drop_duplicates(keep=False)
print(test_dataset.shape)
print(train_dataset.shape)
print(i2f)
correct_classification = 0
for index,row in test_dataset.iterrows():
    # print(index)
    neighbors = get_neighbors(train_dataset, row, 11)
    # print(neighbors)
    class_map = {}
    for neighbor in neighbors:
        # print(index_neighbor)
        class_ = train_labels.loc[train_labels[train_labels.columns[0]] == neighbor]
        class_value = class_.iloc[0].iloc[1]
        if class_value in class_map:
            class_map[class_value]+=1
        else:
            class_map[class_value] = 1
    prediction = (max(class_map, key=class_map.get))
    # predicted_outputs.append(prediction)

    actual = get_actual_result(train_labels, i2f[str(index)])
    # actual = 0
    print(index, prediction, actual, neighbors)
    
    if actual == prediction:
        correct_classification+=1
    
print("accuracy", correct_classification/dataset.shape[0])

















# #todo: read from cmd
# #todo: take input from cmd.
# # print(all_labels)
# dataset = pd.concat([dataset,all_labels.iloc[:,-1:]],axis=1,ignore_index=True)
# test_labels = pd.DataFrame([row for index, row in all_labels.iterrows() if pd.isna(row.iloc[1]) or pd.isnull(row.iloc[1])])
# # print(test_labels)

# test_labels_indexes = [index for index,row in enumerate(test_labels)]
# predicted_outputs = []

# actual_results = {}



