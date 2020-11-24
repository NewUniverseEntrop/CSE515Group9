
import json
import pandas as pd
import numpy as np
import sys,os

# label.csv and similarity_matrix_dtw.npy  should exist in input folder
folder = sys.argv[1]
os.chdir(folder)


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
    
    
seed_vector=np.array([1,0,1,0]).reshape(-1,1)
seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1,1)
with open('f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open('i2f.dump', 'r') as fp:
    i2f = json.load(fp)

class_label_map = {'vattene' : 0, 'combinato' : 1, 'daccordo' : 2}
reverse_dictionary = {v:k for k,v in class_label_map.items()}
labels_df = pd.read_excel('labels.xlsx', names=['file', 'label'], header=None).astype(str)
labels_df['numeric_label'] = labels_df['label'].apply(lambda label : class_label_map[label])
file_label_dict = dict(zip(labels_df.file, labels_df.numeric_label))


class_labels = {0 : [], 1:[], 2:[]}
file_indices = list(i2f.keys())
file_indices = [int(k) for k in file_indices]
for i,row in labels_df.iterrows():
    index = f2i[str(row['file'])]
    label = int(row['numeric_label'])
    class_labels[label].append(index)
    file_indices.remove(index)

print(class_labels)



def getRanksForClass(similarity_matrix, label, class_labels, file_index):
    transformed_similarity_matrix = get_transition_matrix(similarity_matrix, label, class_labels, file_index)
    transformed_similarity_matrix = np.apply_along_axis(column_normalize, axis=0, arr=transformed_similarity_matrix)
    
    seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1, 1)
    for i in class_labels[label]:
        seed_vector[i] = 1
    
    seed_vector[file_index] = 1
    ranks = PPR(transformed_similarity_matrix, seed_vector, max_iterations=100)
    return ranks



ranks_0 = getRanksForClass(similarity_matrix,0,class_labels,file_indices)
ranks_1 = getRanksForClass(similarity_matrix,1,class_labels,file_indices)
ranks_2 = getRanksForClass(similarity_matrix,2,class_labels,file_indices)

ranks_2d = np.array([ranks_0,ranks_1,ranks_2])
ranks_2d = np.take(ranks_2d, file_indices, axis=1)
print(ranks_2d)
result = np.argmax(ranks_2d, axis=0)

n = len(file_indices)
x = 0
for index, i in enumerate(file_indices):
    file = i2f[str(i)]
    label = reverse_dictionary[result[index][0]]
    numberic_label = result[index][0]
    print(str(file)  + " - " +  label +  " - " + str(result[index]) )
    if '_' in file:
        file = file.split('_')[0]
        true_label = file_label_dict[file]
        if(true_label == numberic_label):
            x+=1
    else:
        file = int(file)
        if file<32:
            if numberic_label == 0:
                x+=1
        elif file<300 and file>200:
            if numberic_label == 1:
                x += 1
        elif file > 400:
            if numberic_label == 2:
                x+=1

print("Accuracy : ", float(x/n) * 100)
            

