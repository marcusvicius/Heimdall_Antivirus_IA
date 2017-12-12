from __future__ import print_function

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def get_antivirus_data():
    """Get the antivirus data, from local csv or pandas repo."""
    if os.path.exists("dadosAntivirusIA1.csv"):
        print("-- dadosAntivirusIA.csv found locally")
        df = pd.read_csv("dadosAntivirusIA1.csv", index_col=0)
    else:
        print("-- arquivo nao encontrado")
    return df


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)

df = get_antivirus_data()
print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")
print("* virus types:", df["Classification"].unique(), sep="\n")
print("* program types:", df["Name"].unique(), sep="\n")

df2, targets = encode_target(df, "Classification")
print("* df2.head()", df2[["Target", "Classification"]].head(), sep="\n", end="\n\n")
print("* df2.tail()", df2[["Target", "Classification"]].tail(), sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")

features = list(df2.columns[:3])
print("* features:", features, sep="\n")

le = preprocessing.LabelEncoder()
le.fit(df["Name"].unique())
print(le.transform(df2.columns[2:2]) )
print(le);
#print(le.transform(df2.columns[1:2])) 
#print(df2)

y = df2["Target"]
X = df2[features]



dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)
visualize_tree(dt, features)
get_code(dt, features, targets)