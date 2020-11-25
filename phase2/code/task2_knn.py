import heapq
import os

import pandas as pd
import numpy as np
import pickle
from math import sqrt
import sys
import json
import ast
import functools

folder = sys.argv[1]
os.chdir(folder)
vecoption = 'tfidf'
k = 9
option = 'pca'

with open('f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open('i2f.dump', 'r') as fp:
    i2f = json.load(fp)


def get_labeled_files_to_classes_map(df):
    return  dict(zip(df.file, df.label))

class_label_map = {'vattene' : 0, 'combinato' : 1, 'daccordo' : 2}

train_labels = pd.read_excel('labels.xlsx', header=None, names=['file', 'label']).astype(str)
train_labels['numeric_label'] = train_labels['label'].apply(lambda label : class_label_map[label])
file_label_dict = dict(zip(train_labels.file, train_labels.numeric_label))
labeled_files_to_classes_map = get_labeled_files_to_classes_map(train_labels)

labeled_files_indices = set([f for f in list(labeled_files_to_classes_map.keys())])
unlabeled_files_indices = set(f2i.keys()).difference(labeled_files_indices)

labeled_files_indices = [ int(f2i[f]) for f in labeled_files_indices ]
unlabeled_files_indices = [ int(f2i[f]) for f in unlabeled_files_indices ]
# dataset = pd.read_csv(folder+'/train_'+option+'_'+vecoption+'.csv', header=None)
#
# train_dataset = pd.DataFrame()

distance_matrix = np.load('distance_dtw.matrix.npy')

n = len(unlabeled_files_indices)
x = 0
for file_index in unlabeled_files_indices:
    
    distances = distance_matrix[file_index]
    map = {}
    for index, distance in enumerate(distances):
        if index in labeled_files_indices:
            map[index] = distance
    print(map)
    k_keys_sorted = heapq.nsmallest(k, map,key=map.__getitem__)
    print(k_keys_sorted)
    label_map = {}
    for f_index in k_keys_sorted:
        label = labeled_files_to_classes_map[i2f[str(f_index)]]
        if label in label_map:
            label_map[label]+=1
        else:
            label_map[label] = 1
    print([i2f[str(i)] for i in k_keys_sorted])
    prediction = (max(label_map, key=label_map.get))
    print(class_label_map[prediction], i2f[str(file_index)])
    file = i2f[str(file_index)]
    numeric_label = class_label_map[prediction]
    if '_' in file:
        file = file.split('_')[0]
        true_label = file_label_dict[file]
        if(true_label == numeric_label):
            x+=1
    else:
        file = int(file)
        if file<32:
            if numeric_label == 0:
                x+=1
        elif file<300 and file>200:
            if numeric_label == 1:
                x += 1
        elif file > 400:
            if numeric_label == 2:
                x+=1
print("Accuracy : ", float(x/n) * 100)




