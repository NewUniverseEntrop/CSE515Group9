import pandas as pd

# Split a dataset based on an attribute and an attribute value
def test_split(column, value, dataset):
    left, right = list(), list()
    for index,row in dataset.iterrows():
        # print(row.iloc[column])
        if row.iloc[column] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row.iloc[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row.iloc[-1] for index,row in dataset.iterrows() ))
    print(class_values)    
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    # print(dataset.shape[1]-1)
    for column in range(0,dataset.shape[1]-1):
        for index,row in dataset.iterrows():
            # print(row)
            groups = test_split(column, row.iloc[column], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = column, row.iloc[column], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row.iloc[-1] for index,row in group.iterrows()]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    left = pd.DataFrame(left)
    right = pd.DataFrame(right)
    del(node['groups'])
    # check for a no split
    if left.shape[0] == 0 or right.shape[0] == 0:
        node['left'] = node['right'] = to_terminal(left.append(right))
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


# Make a prediction with a decision tree
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


#todo: read from cmd
dataset = pd.read_csv('C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\3_class_gesture_data\\train_pca_tfidf.csv', header=None)
all_labels = pd.read_excel('C:\\Users\\manuh\\Documents\\MWDB\\Phase3_\\CSE515Group9\\phase2\\3_class_gesture_data\\labels.xlsx', header=None) 
#todo: take input from cmd.
dataset = pd.concat([dataset,all_labels.iloc[:,-1:]],axis=1,ignore_index=True)
test_labels = pd.DataFrame([row for index, row in all_labels.iterrows() if pd.isna(row.iloc[1]) or pd.isnull(row.iloc[1])])

train_dataset = dataset[dataset.iloc[:,-1].notna()]
print(train_dataset.shape)

tree = build_tree(train_dataset,3,1)
print_tree(tree)


test_labels_indexes = [index for index,row in enumerate(test_labels)]
predicted_outputs = []

for index,row in test_labels.iterrows():
    prediction = predict(tree,dataset.iloc[index])
    predicted_outputs.append(prediction)

    print(index, prediction)
    # break