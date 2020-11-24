import pandas as pd
import numpy as np
import pickle
from math import sqrt


def get_neighbors(train, test_row_index, num_neighbors, test_label_indexes):
    distances = list()
    for index,train_row in train.iterrows():
        dist = euclidean_distance(train.iloc[test_row_index], train_row)
        distances.append((index, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    count=0
    i=0
    while(count<num_neighbors):
        if distances[i][0] not in test_label_indexes:
            neighbors.append(distances[i][0])
            count+=1
        i+=1
    return neighbors


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1.iloc[i] - row2.iloc[i])**2
    return sqrt(distance)


#todo: read from cmd
dataset = pd.read_csv('C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\3_class_gesture_data\\train_pca_tfidf.csv', header=None)
all_labels = pd.read_excel('C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\3_class_gesture_data\\labels.xlsx', header=None) 
#todo: take input from cmd.
# print(all_labels)
dataset = pd.concat([dataset,all_labels.iloc[:,-1:]],axis=1,ignore_index=True)
test_labels = pd.DataFrame([row for index, row in all_labels.iterrows() if pd.isna(row.iloc[1]) or pd.isnull(row.iloc[1])])
# print(test_labels)

test_labels_indexes = [index for index,row in enumerate(test_labels)]
predicted_outputs = []

for index,row in test_labels.iterrows():
    # print(index)
    neighbors = get_neighbors(dataset, index, 10, test_labels_indexes)
    # print(neighbors)
    class_map = {}
    # print(dataset.iloc[0])
    # print(dataset.iloc[0].iloc[-1])
    for neighbor in neighbors:
        if dataset.iloc[neighbor].iloc[-1] in class_map:
            class_map[dataset.iloc[neighbor].iloc[-1]]+=1
        else:
            class_map[dataset.iloc[neighbor].iloc[-1]] = 1
    prediction = (max(class_map, key=class_map.get))
    predicted_outputs.append(prediction)

    print(index, prediction)
    # break
    


