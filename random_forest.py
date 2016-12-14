"""This file uses a random forest, a bunch of individual unpruned decision trees with access to only
some of the features. Final classification is then done by majority vote, to remove overfitting
noise and make the model extremely robust.

There's one interesting addendum to this: in a random forest decision tree, making a split is only
done with a subset of the features available at every step. This way, strong predictors don't
dominate every tree they appear in and overly couple the trees.

Team: Jocelyn, Kevin, Nicholas
Author: Nicholas
"""

from itertools import combinations
from random import sample
from random_tree import RandomDecisionTree

import pandas as pd


class RandomForestClassifier:
    """A random forest classifier."""
    def __init__(self, n, b, n_ratio=True, b_ratio=True):
        """Initializes a bare random forest classifier. N is the number of elements to bootstrap from the
        training data, or a ratio of the total data size if n_ratio is True. B is the number of
        trees to use when classifying. If b_ratio is True, b will instead be a ratio of the total
        data input dimensionality.
        """
        self.n = n
        self.n_ratio = n_ratio
        self.b_ratio = b_ratio
        self.b = b
        self.trees = []

    @classmethod
    def get_all_subsets(self, items):
        """Returns a list of every subset of the given input."""
        subsets = []
        for i in range(1, len(items)+1):
            subsets += [combo for combo in combinations(items, i)]
        return subsets

    @classmethod
    def get_random_samples(cls, choices, n):
        """Draws n different subsets of choices and returns them as a list."""
        return sample(cls.get_all_subsets(choices), n)

    def fit(self, X, y):
        """Fits a random forest classifier on X mapping to Y, a vector with the same height but
        classes instead of predictors."""
        
        # get correct number of trees
        tree_num = self.b if not self.b_ratio else int(self.b * X.shape[1])

        # get the right number of samples of features
        feature_samples = self.get_random_samples(X.columns, tree_num)
        # get the right number of samples of data
        bootstrap_size = self.n if not self.n_ratio else int(self.n * X.shape[0])
        bootstraps = [sample(range(X.shape[0]), bootstrap_size) for i in range(tree_num)]

        data_subsets = [(X.loc[bootstraps[i], feature_samples[i]], y.loc[bootstraps[i], :])
                        for i in range(tree_num)]
        self.trees = [RandomDecisionTree() for i in range(tree_num)]
        for i in range(tree_num):
            self.trees[i].fit(*data_subsets[i])  # fit that particular one

    def predict(self, X_new):
        """Uses majority voting with a random forest to predict y."""
        votes = [tree.predict(X_new) for tree in self.trees]  # get predictions
        return max(votes, key=lambda x: votes.count(x))  # get most common and return it



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
y = df.loc[:, ["Class Label"]]

X["Body Temperature"] = X["Body Temperature"].apply(lambda x: 1 if x == "warm-blooded" else 0)
for col in X.columns[1:]:
    X[col] = X[col].apply(lambda x: 1 if x == "yes" else 0)

X_training = X.loc[:7, :]
X_test = X.loc[7:, :]

y_training = y.loc[:7, ["Class Label"]]
y_test = y.loc[7:, ["Class Label"]]

y["Class Label"] = pd.DataFrame(y["Class Label"].apply(lambda x: 1 if x == "yes" else 0))


rf = RandomForestClassifier(1, 1)
rf.fit(X_training, y_training)
print(rf.predict(X_test))
print(y_test)
