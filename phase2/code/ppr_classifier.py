
import json
import pandas as pd
import numpy as np
import sys,os
os.chdir('/Users/xavi/MWDB/xin/CSE515Group9/phase2/3_class_gesture_data')


def get_transition_matrix(adj_matrix,class_label,labels,indices):
    similarity_matrix = np.copy(adj_matrix)
    objects_same_label = labels[class_label]
    print(objects_same_label)
    for i in range(len(similarity_matrix)):
        if i  not in objects_same_label :
            for index in indices:
                similarity_matrix[index][i] = 0
                similarity_matrix[i][index] = 0
    return similarity_matrix
    
def getNearestNeightbours(arr,k=2):
    if np.sum(arr) == 0:
        return arr
    indices = np.argsort(arr)[-k:]
    for i,x in enumerate(arr):
        if i not in indices:
            arr[i] = 0
    return arr/np.sum(arr)

def column_normalize(arr):
    if np.sum(arr) == 0:
        return arr
    return arr/np.sum(arr)

def PPR(adjacency_matrix, seed_vector, max_iterations=100, beta_probabilty=0.15):
    n = adjacency_matrix.shape[0]
    seed_vector = seed_vector / np.sum(seed_vector)
    x = np.full((n, 1), 1 / n)
    
    for i in range(max_iterations):
        x_prev = x
        a = (1 - beta_probabilty) * np.matmul(adjacency_matrix, x)
        b = seed_vector * beta_probabilty
        x = np.add(a, b)
        sum = np.sum(np.abs(x - x_prev))
        if (sum < 1.0e-6):
            print(i, sum)
            break
    return x

similarity_matrix = np.load('similarity_matrix_dtw.npy')
for i in range(len(similarity_matrix)):
    similarity_matrix[i][i] = 0
k = 5
seed_vector=np.array([1,0,1,0]).reshape(-1,1)
seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1,1)
with open('f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open('i2f.dump', 'r') as fp:
    i2f = json.load(fp)

labels_df = pd.read_csv('labels.csv')
labels_df = labels_df.fillna(-1)
classified_labels = labels_df[labels_df['label'] != -1]
print(classified_labels)

unclassified_label = labels_df[labels_df['label'] == -1]
print(unclassified_label)


class_labels = {0 : [], 1:[], 2:[]}

for i,row in classified_labels.iterrows():
    index = f2i[str(int(row['file']))]
    label = int(row['label'])
    class_labels[label].append(index)
# for i in range(31):
#     class_labels[0].append(i)
# for i in range(31,62):
#     class_labels[1].append(i)
# for i in range(62,93):
#     class_labels[2].append(i)

# class_labels[2].remove(69)
print(class_labels)
file_indices = [0,92]

#class1
transformed_similarity_matrix = get_transition_matrix(similarity_matrix, 0, class_labels, file_indices)
transformed_similarity_matrix= np.apply_along_axis(column_normalize,axis=0,arr=transformed_similarity_matrix)

seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1,1)
for i in class_labels[0]:
    seed_vector[i] = 1

for unclassified_gesture in file_indices:
    seed_vector[unclassified_gesture] = 1
# seed_vector[file_index] = 1
ranks = PPR(transformed_similarity_matrix,seed_vector,max_iterations=100)
score1 = ranks[file_indices]
# print(ranks[file_indices])
for file_index in file_indices:
    print(ranks[file_index])

#class2
transformed_similarity_matrix = get_transition_matrix(similarity_matrix, 1, class_labels, file_indices)
transformed_similarity_matrix= np.apply_along_axis(column_normalize,axis=0,arr=transformed_similarity_matrix)

seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1,1)
for i in class_labels[1]:
    seed_vector[i] = 1

for unclassified_gesture in file_indices:
    seed_vector[unclassified_gesture] = 1
# seed_vector[file_index] = 1
ranks = PPR(transformed_similarity_matrix,seed_vector,max_iterations=100)
for file_index in file_indices:
    print(ranks[file_index])


#class3
transformed_similarity_matrix = get_transition_matrix(similarity_matrix, 2, class_labels, file_indices)
transformed_similarity_matrix= np.apply_along_axis(column_normalize,axis=0,arr=transformed_similarity_matrix)

seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1,1)
for i in class_labels[2]:
    seed_vector[i] = 1

for unclassified_gesture in file_indices:
    seed_vector[unclassified_gesture] = 1
# seed_vector[file_index] = 1
ranks = PPR(transformed_similarity_matrix,seed_vector,max_iterations=100)
for file_index in file_indices:
    print(ranks[file_index])















def getRanksForClass(similarity_matrix,label,class_labels,file_index):
    transformed_similarity_matrix = get_transition_matrix(similarity_matrix, label, class_labels, file_index)
    transformed_similarity_matrix = np.apply_along_axis(column_normalize, axis=0, arr=transformed_similarity_matrix)
    
    seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1, 1)
    for i in class_labels[label]:
        seed_vector[i] = 1
    
    seed_vector[file_index] = 1
    ranks = PPR(transformed_similarity_matrix, seed_vector, max_iterations=100)
    return ranks