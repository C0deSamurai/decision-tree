"""This file implements a subclass of DecisionTree that has one small caveat: when it makes a split,
it chooses a random subset of the features it has to make a split with, thereby reducing the tree's
tendency to be dominated by a few strong predictors.

Team: Jocelyn, Kevin, Nicholas
Author: Nicholas
"""


from decision_tree import DecisionTree
from split import Split
from random import sample


class RandomDecisionTree(DecisionTree):
    """A random forest decision learner."""
    def __init__(self):
        """Initializes a RandomDecisionTree ready for fitting."""
        super().__init__()

    def gen_splits(self, data, classes):
        """Generates all possible splits using only the square root of the number of available
        features."""
        splits = []
        for predictor_name in sample(data.columns, int(len(data.columns) ** (0.5))):
            pred = data[predictor_name]
            # generate every possible split cutoff that would mean something
            cutoffs = list(pred)
            cutoffs = [c for c in cutoffs if c != max(cutoffs)]  # remove topmost element
            splits += [Split(predictor_name, cutoff) for cutoff in cutoffs]
        return splits

    def fit(self, X, y):
        """Does no pruning, just the custom feature bagging. Fits the tree to predict the classes in
        y given the features in X."""
        super().raw_fit(X, y)
