"""This file just does some testing of the decision tree program."""


import pandas as pd

from decision_tree import DecisionTree
from tree import Tree
from node import Node
from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()


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

iris_x_train, iris_x_test, iris_y_train, iris_y_test = \
    train_test_split(iris.data, iris.target, test_size=0.2)

iris_x_train = pd.DataFrame(iris_x_train, columns=iris.feature_names)
iris_x_test = pd.DataFrame(iris_x_test, columns=iris.feature_names)
iris_y_train = pd.DataFrame({'type': iris_y_train})
iris_y_test = pd.DataFrame({'type': iris_y_test})

dt = DecisionTree()
dt.raw_fit(iris_x_train, iris_y_train)
print(iris_x_test.apply(dt.predict))
print(iris_y_test)
