import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def get_numeric_rep(df, y_cols):
    '''
    Just encoding
    '''
    mapped_data = {}
    for y_col in y_cols:
        label_dict = {lbl: ind for ind, lbl in enumerate(
            df[y_col].unique())}
        df[y_col] = df[y_col].map(label_dict)
        mapped_data[y_col] = (label_dict)
    return mapped_data, df


def print_decision_tree(tree, feature_names=None, offset_width='    '):
    '''Textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1,f2,f3,... if not specified
    offset_width: a string of offset of the conditional block'''

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    if feature_names is None:
        features = ['f%d' % i for i in tree.tree_.feature]
    else:
        features = [feature_names[i] for i in tree.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0):
        offset = offset_width * depth
        if (threshold[node] != -2):
            print(
                offset + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
            if left[node] != -1:
                recurse(left, right, threshold,
                        features, left[node], depth + 1)
            print(offset + "} else {")
            if right[node] != -1:
                recurse(left, right, threshold,
                        features, right[node], depth + 1)
            print(offset + "}")
        else:
            print(offset + "return " + str(value[node]))

    recurse(left, right, threshold, features, 0, 0)


pickle_in = open("dataset.pickle", "rb")
dataset = pickle.load(pickle_in)

cols = list(dataset)
target_data = dataset[[cols[0]]]  # First column is the target variable here

tree = DecisionTreeClassifier()

# one_hot_data = pd.get_dummies(data[['A','B','C']])

df = pd.DataFrame(dataset[cols[1:]])

mapped_data, one_hot_data = get_numeric_rep(df, cols)

__, target_data = get_numeric_rep(target_data, [cols[0]])


tree.fit(one_hot_data, target_data)
print_decision_tree(tree, feature_names=cols, offset_width='    ')
