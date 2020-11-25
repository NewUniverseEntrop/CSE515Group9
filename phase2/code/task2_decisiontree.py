import pandas as pd
import pickle
import json
import sys

def test_split_attribute(column, value, dataset):
    left, right = list(), list()
    for index,row in dataset.iterrows():
        if row.iloc[column] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def compute_gini_index(groups, classes):
    num_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row.iloc[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / num_instances)
    return gini

def get_split_pts(dataset):
    class_values = list(set(row.iloc[-1] for index,row in dataset.iterrows() ))
    print(class_values)    
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for column in range(0,dataset.shape[1]-1):
        for index,row in dataset.iterrows():
            groups = test_split_attribute(column, row.iloc[column], dataset)
            gini = compute_gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = column, row.iloc[column], gini, groups
    return {'index':best_index, 'value':best_value, 'groups':best_groups}

def get_term_node(group):
    outcomes = [row.iloc[-1] for index,row in group.iterrows()]
    return max(set(outcomes), key=outcomes.count)

def split_till_term(node, max_depth, min_size, depth):
    left, right = node['groups']
    left = pd.DataFrame(left)
    right = pd.DataFrame(right)
    del(node['groups'])
    if left.shape[0] == 0 or right.shape[0] == 0:
        node['left'] = node['right'] = get_term_node(left.append(right))
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_term_node(left), get_term_node(right)
        return
    if len(left) <= min_size:
        node['left'] = get_term_node(left)
    else:
        node['left'] = get_split_pts(left)
        split_till_term(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = get_term_node(right)
    else:
        node['right'] = get_split_pts(right)
        split_till_term(node['right'], max_depth, min_size, depth+1)

def build_DTree(train, max_depth, min_size):
    root = get_split_pts(train)
    split_till_term(root, max_depth, min_size, 1)
    return root

def print_DTree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_DTree(node['left'], depth+1)
        print_DTree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

def predict(node, row):
    if row.iloc[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

#save tree
def save_model(tree, max_depth, min_size):
    with open('decision_tree'+str(max_depth)+'_'+str(min_size)+'.obj', 'w') as f:
        pickle.dump(tree, f)

def get_actual_result(train_labels, index):
    index_split = index.split("_")[0]
    class_ = train_labels.loc[train_labels[train_labels.columns[1]] == index_split]
    class_value = class_.iloc[0].iloc[2]
    return class_value


folder = sys.argv[1]
vecoption = sys.argv[2]  # tf, tf-idf
option = sys.argv[3]     # pca, svd, nfm, lda

with open(folder+'/f2i.dump', 'r') as fp:
    f2i = json.load(fp)
with open(folder+'/i2f.dump', 'r') as fp:
    i2f = json.load(fp)

train_labels = pd.read_excel(folder+'/labels.xlsx', header=None,names=['gesture','label']).astype(str)
dataset = pd.read_csv(folder+'/train_'+option+'_'+vecoption+'.csv', header=None)
train_dataset = pd.DataFrame()



for index, row in train_labels.iterrows():
    index_in_dataset = f2i[str(row.iloc[0])]
    train_row = dataset.iloc[index_in_dataset]
    train_dataset = train_dataset.append(train_row)

print(train_dataset.head(40))
test_dataset = pd.concat([dataset,train_dataset]).drop_duplicates(keep=False)


train_dataset = train_dataset.reset_index()
train_labels= train_labels.reset_index()


train_dataset = pd.concat([train_dataset,train_labels.iloc[:,-1:]], axis=1, ignore_index=True)
train_dataset = train_dataset.set_index(train_dataset.columns[0])
print(train_dataset.head(40))

tree = build_DTree(train_dataset,4,1)
print_DTree(tree)
# save_model(tree,3,1)


# print(i2f)
correct_classification = 0
for index,row in test_dataset.iterrows():
    prediction = predict(tree, row)
    actual = get_actual_result(train_labels, i2f[str(index)])
    # actual = 0
    # print(index, prediction, actual)
    
    if actual == prediction:
        correct_classification+=1
    
print("accuracy", correct_classification/test_dataset.shape[0])
