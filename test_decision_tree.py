"""This file just does some testing of the decision tree program."""


import pandas as pd

from decision_tree import DecisionTree
from tree import Tree
from node import Node


mat = [["human", "warm-blooded", "yes", "no", "no", "yes"],
       ["pigeon", "warm-blooded", "no", "no", "no", "no"],
       ["elephant", "warm-blooded", "yes", "yes", "no", "yes"],
       ["leopard shark", "cold-blooded", "yes", "no", "no", "no"],
       ["turtle", "cold-blooded", "no", "yes", "no", "no"],
       ["penguin", "cold-blooded", "no", "no", "no", "no"],
       ["eel", "cold-blooded", "no", "no", "no", "no"],
       ["dolphin", "warm-blooded", "yes", "no", "no", "yes"],
       ["spiny anteater", "warm-blooded", "no", "yes", "yes", "yes"],
       ["gila monster", "cold-blooded", "no", "yes", "yes", "no"]]

df = pd.DataFrame(mat, columns=["Name",
                                "Body Temperature",
                                "Gives Birth",
                                "Four-legged",
                                "Hibernates",
                                "Class Label"])


X = df.loc[:, ["Body Temperature", "Gives Birth", "Four-legged", "Hibernates"]]
y = df.loc[:, "Class Label"]

X["Body Temperature"] = X["Body Temperature"].apply(lambda x: 1 if x == "warm-blooded" else 0)
for col in X.columns[1:]:
    X[col] = X[col].apply(lambda x: 1 if x == "yes" else 0)

y = pd.DataFrame(y.apply(lambda x: 1 if x == "yes" else 0))


tree = Tree(Node('a',
                 Node('b', None,
                      Node('c', None,
                           Node('d', None, None))),
                 Node('e', None, None)))


def print_tree(t):
    print('\n'.join([repr(x) for x in list(t)]))

#print()
#print(tree)
#tree.set_child(5, 0, Node('f', None, None))
#print()
#print(tree)

x_pred = X.iloc[0]

dt = DecisionTree()
dt.raw_fit(X, y)
dt.predict(x_pred)
