
# run phase 2 task3 to generate similarity matrix before running PPR
#   gesturedisc.py /Users/xavi/MWDB/xin/CSE515Group9/phase2/3_class_gesture_data tfidf dtw 3 svd
# task1.py /Users/xavi/MWDB/xin/CSE515Group9/phase2/3_class_gesture_data 2 10 1,2,3,4,5
# task1.py cmd parameters :  dir_path k m and user selected gestures


import json
import pandas as pd
import numpy as np
import sys,os
import matplotlib.pyplot as plt


folder = sys.argv[1]
os.chdir(folder)
k = int(sys.argv[2])
m = int(sys.argv[3])
user_selected_gestures = list(sys.argv[4].split(','))

def visualise_component(gesture,component):
    filename = component + '/' + str(gesture) + '.csv'
    df = pd.read_csv(filename,header=None).T
    df_norm = df.copy()
    for column in df_norm.columns:
        min = df_norm[column].min()
        max = df_norm[column].max()
        if min == max:
            df_norm[column].values[:] = 0
        else:
            df_norm[column] = (2*(df_norm[column] - min)) / (max - min) - 1
    series = df_norm[df_norm.columns[0]].to_numpy()
    for column in df_norm.columns:
        plt.plot(df_norm[column].to_numpy(), label=column)
    # plt.plot(series,label = '0')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    

def visualise_gesture(gesture):
    for component in ['W','X','Y','Z']:
        visualise_component(gesture,component)
        break


def visualise_gestures(m_gestures):
    for gesture in m_gestures:
        visualise_gesture(gesture)
        break
        
def getNearestNeightbours(arr,k=2):
    print(arr)
    if np.sum(arr) == 0:
        return arr
    indices = np.argsort(arr)[-k:]
    for i,x in enumerate(arr):
        if i not in indices:
            arr[i] = 0
    return arr/np.sum(arr)



def PPR(adjacency_matrix,seed_vector,max_iterations=100,beta_probabilty=0.15):
    n = adjacency_matrix.shape[0]
    seed_vector = seed_vector/np.sum(seed_vector)
    x = np.full((n,1),1/n)
    print(x)
    print(seed_vector.shape)
    print("seed")
    
    for i in range(max_iterations):
        x_prev = x
        a = (1-beta_probabilty)*np.matmul(adjacency_matrix,x)
        b = seed_vector*beta_probabilty
        x = np.add(a,b)
        print(x)
        sum = np.sum(np.abs(x-x_prev))
        if(sum < 1.0e-6):
            print(i,sum)
            break
    return x
    

similarity_matrix = np.transpose(np.array([[0,0.03,0.02,0.01],[0.9,0,0,0],[0.7,0,0,0],[0.5,0,0,0]]))
# similarity_matrix = np.transpose(np.array([[0,0.3,0,0],[0,0,0,0],[0,0.9,0,0],[0,0,0.1,0]]))
# similarity_matrix = np.transpose(np.array([[0,0.4,0.2,0.9],[0.4,0,0.3,0.06],[0.2,0.3,0,0.7],[0.9,0.06,0.7,0]]))
similarity_matrix = np.load('similarity_matrix_dtw.npy')
for i in range(len(similarity_matrix)):
    similarity_matrix[i][i] = 0
print(similarity_matrix)

# k = 4
seed_vector= np.array([1,0,1,0]).reshape(-1,1)
seed_vector = np.zeros(similarity_matrix.shape[0]).reshape(-1,1)
with open('f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open('i2f.dump', 'r') as fp:
    i2f = json.load(fp)

for gesture in user_selected_gestures:
    seed_vector[f2i[gesture]] = 1

# seed_vector[42] = 1
# seed_vector[44] = 1
# seed_vector[55] = 1
# seed_vector[51] = 1
# seed_vector[60] = 1

print(seed_vector)
# m = 10

transformed_similarity_matrix= np.apply_along_axis(getNearestNeightbours,axis=0,arr=similarity_matrix,k=k)
print(transformed_similarity_matrix)
ranks = PPR(transformed_similarity_matrix,seed_vector,max_iterations=100)
print(ranks)
ranks = ranks.reshape(-1)
print("output")
print(ranks.argsort()[-m:][::-1])
output = ranks.argsort()[-m:][::-1]
# visualise_gestures(output)



