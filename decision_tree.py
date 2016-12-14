
"""This class creates a decision tree classifier for a given dataset."""

from copy import deepcopy


import pandas as pd
from node import Node
from split import Split
from tree import Tree


class DecisionTree:
    """Represents a decision tree classifier."""

    def __init__(self):
        """Creates a decision tree that can be fit to data."""
        self.tree = None

    def __str__(self):
        """Ignores DataFrames, only prints splits."""
        new_tree = deepcopy(self.tree)
        for node in [n for n in list(new_tree) if n is not None]:
            if len(node.val) == 1 or node.val[1] is None:
                node.val = "<data>"
            else:
                node.val = str(node.val[1])
        return str(new_tree)

    @classmethod
    def gini_vector(cls, col):
        """Given a single input Series with a single column filled with the digits 0-n, computes the
    Gini impurity."""
        impurity = 0
        n = len(set(col))
        if n <= 1:  # sets without any elements or with a single class are pure by default
            return 0

        for i in range(n):
            p_class = sum(col.map(lambda x: 1 if x == i else 0)) / len(col)
            impurity += p_class ** 2

        return 1 - impurity

    @classmethod
    def gini(cls, classes):
        """Computes the Gini impurity of the given data, a class vector or matrix in 0's and 1's or any
    additional amount of classes.
        """
        return classes.apply(cls.gini_vector).sum()

    def gen_splits(self, data, classes):
        """Given input data and a corresponding matrix of classes, returns a list of Split objects
        corresponding to all the possible splits of that dataset at that time."""
        splits = []
        for predictor_name in data.columns:
            pred = data[predictor_name]
            # generate every possible split cutoff that would mean something
            # also, remove duplicates
            cutoffs = list(set(pred))
            cutoffs = [c for c in cutoffs if c != max(cutoffs)]  # remove topmost element
            splits += [Split(predictor_name, cutoff) for cutoff in cutoffs]
        return splits

    def execute_split(self, data, classes, split):
        """Given data with associated classes and split data, returns a tuple ((data_false, classes_false)
        (data_true, classes_true)) with the entries that are false and the entries that satisfy the
        split.
        """

        # generate indices of cutoffs instead of just the data to preserve class data
        good_indices = [i for i in range(data.shape[0]) if split.split(i, pd.DataFrame(data), True)]
        bad_indices = [i for i in range(data.shape[0]) if i not in good_indices]

        good = (data.iloc[good_indices, :], classes.iloc[good_indices, :])
        bad = (data.iloc[bad_indices, :], classes.iloc[bad_indices, :])
        return (bad, good)

    def test_split(self, data, classes, split):
        """Given data with associated classes and a split, returns the sum of the Gini impurity of
        the child nodes of this split if it is effected."""

        left, right = self.execute_split(data, classes, split)
        return self.gini(left[1]) + self.gini(right[1])

    def not_overfit(self, data, classes, split, k=0.05):
        """This method can be overridden or replaced to change how much the fitting algorithm prunes the
        tree. The default is to require that at least a fraction of the input be separated (the
        defaut is 5%)."""
        left, right = self.execute_split(data, classes, split)
        
        return k <= left[0].shape[0] / data.shape[0] <= 1 - k

    def create_split(self, pos, prune_func):
        """Given a position in the decision tree (1-based breadth-first indexing), first checks to see if
        the data is pure. If it is, then it sets both children to None and stops. Otherwise, it
        computes the split that minimizes the Gini impurity for each child node and executes it,
        creating two new children that are the results of the split (left for false, right for
        true).

        Prune_func is the pruning function to determine whether a split is valid.
        """
        # print("Creating split at position {}".format(pos))

        node = self.tree[pos-1]
        node_val = node.val[0]  # the tuple (data, classes)
        curr_gini = self.gini(node_val[1])

        if curr_gini == 0:  # data is pure
            self.tree.set_child(pos, 0, None)
            self.tree.set_child(pos, 1, None)  # make this a leaf
            return None  # we're done here!
        else:  # generate possible splits
            splits = self.gen_splits(*node_val)
            splits = [split for split in splits if prune_func(*node_val, split)]
            # print(splits)
            if len(splits) == 0:  # no splits that separate into at least one class
                self.tree.set_child(pos, 0, None)
                self.tree.set_child(pos, 1, None)
                return None
            # find the best one
            best = min(splits, key=lambda split: self.test_split(*node_val, split))
            # print(sorted(splits, key=lambda split: self.test_split(*node_val, split))[:5])
            if len(node.val) == 1:  # no split in the node yet
                node.val.append(best)
            else:  # overwrite a split
                node.val[1].append(best)
            # execute it
            left, right = self.execute_split(*node_val, best)
            # print(best)
            # print(left[0].shape[0], right[0].shape[0])

            self.tree.set_child(pos, 0, Node([left], None, None))
            self.tree.set_child(pos, 1, Node([right], None, None))

    def recursively_create_splits(self, pos, prune_func):
        """For the current Node, recursively splits all of its children (generating them as it goes) until
        all of the leaves are pure, returning None. Prune_func is the function to reduce
        overfitting.

        """
        if pos > len(list(self.tree)) or self.tree[pos-1] is None:  # we stop here
            return None
        else:  # create two children and recurse to split them
            self.create_split(pos, prune_func)
            self.recursively_create_splits(pos * 2, prune_func)  # left child
            self.recursively_create_splits(pos * 2 + 1, prune_func)  # right child
            return None

    def fit(self, X, y, pruning_func=None):
        """Fits a given input matrix to a given output vector using a decision tree. X should be a DataFrame
        of numerical variables with any index or column names. y should be a vector or matrix with
        the same height as X, as many columns as classes to predict, and each column should be a
        list of 0's and 1's for a given class. The pruning_func parameter can be filled with any
        function that takes in three arguments (data, classes, split) and returns True if that split
        is allowed and False otherwise. The default, specified with None, is to only use splits that
        separate 5% of the input.

        """
        # start with just the entire dataset at the root, with no split
        # I'm going to represent a single node of a tree as a value [(data, classes), split]
        # where data and classes are "what's left" and split is the Split function that gives the
        # left and right children, or None if the data is pure or the splits have stopped
        if pruning_func is None:  # use default
            pruner = self.not_overfit
            
        self.tree = Tree(Node([(X, y)], None, None))
        self.recursively_create_splits(1, prune_func=pruner)  # pretty anticlimactic
        return None

    def __predict_vec(self, X):
        """Given an input vector with the required number of predictors, returns an output vector
        y_hat representing the predicted classes for all of the classes it was fit to."""

        if self.tree is None:  # nothing to predict with!
            raise ValueError("Must fit to a model before prediction!")

        curr_pos = 1
        while True:  # loop until something is returned
            curr_node = self.tree[curr_pos-1]
            curr = curr_node.val
            if len(curr) == 1 or curr[1] is None:  # no split, so must be pure or close
                # return the mean of the classes, rounding to integers
                # this way, even with non-pure pruned leaves you still get a result
                if not hasattr(curr[0][1], "__len__"):  # single number
                    return curr[0][1]  # we're done
                else:  # lots of values
                    if len(curr[0][1].shape) > 1:  # 2D array, so use array functions
                        return curr[0][1].mean().round()
                    else:  # just get the mean and use basic round, not a method
                        return int(curr[0][1].mean())
            else:  # go down to the correct child and try again
                if curr[1].split(0, X, use_iloc=True):  # split with the input data

                    curr_pos = curr_pos * 2 + 1  # right child

                else:  # left child, because the Split returned False
                    curr_pos *= 2

            # now, just repeat until we get to a leaf!

    def predict(self, X):
        """For every row in X, predicts the classes and returns a matrix Y representing the
        predicted classes."""
        if len(X.shape) == 1:  # single row
            return self.__predict_vec(X)

        return X.apply(self.__predict_vec, axis=1)

    def __score_col(self, col1, col2):
        """Returns the accuracy score for two Series."""
        correct = 0
        for i in range(len(col1)):
            if col1.iloc[i] == col2.iloc[i]:
                correct += 1
        return correct / len(col1)

    def score(self, X, true_y):
        """Gets the accuracy score of the classifier given the X as input and with true_y as the
        actual values. If there are multiple classes, returns an iterable of each accuracy score."""

        preds = self.predict(X)
        if len(preds.shape) >= 2 and preds.shape[1] > 1:  # multiple classes
            scores = []
            for colname in preds:
                scores.append(self.__score_col(preds[colname], true_y[colname]))
            return scores
        else:  # just one class
            if isinstance(preds, pd.DataFrame):  # convert to Series
                converted_pred = preds.iloc[:, 0]
            else:
                converted_pred = preds

            if isinstance(true_y, pd.DataFrame):  # convert to Series
                converted_true = true_y.iloc[:, 0]
            else:
                converted_true = true_y

            return self.__score_col(converted_pred, converted_true)
