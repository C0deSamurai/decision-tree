
"""This class creates a decision tree classifier for a given dataset."""

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
        return str(self.tree)

    @classmethod
    def gini(cls, classes):
        """Computes the Gini impurity of the given data, a class vector or matrix in 0's and 1's.
        """
        if classes.shape[1] == 1:  # only one entry
            if len(classes) == 0:
                return 0  # the empty list is pure!

            p = len([x for x in classes.iloc[:, 0] if x == 1]) / len(classes)
            return 1 - (1 - p) ** 2 - p ** 2
        impurity = 0
        for class_name in classes:
            class_ = classes[class_name]
            p_class = sum(class_.map(lambda x: 1 if x else 0)) / len(class_)
            impurity += p_class ** 2

        return 1 - impurity  # returns 0 if the data is pure and 0.5 if it is perfectly mixed

    def gen_splits(self, data, classes):
        """Given input data and a corresponding matrix of classes, returns a list of Split objects
        corresponding to all the possible splits of that dataset at that time."""
        splits = []
        for predictor_name in data.columns:
            pred = data[predictor_name]
            # generate every possible split cutoff that would mean something
            cutoffs = list(pred)
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

    def create_split(self, pos):
        """Given a position in the decision tree (1-based breadth-first indexing), first checks to see if
        the data is pure. If it is, then it sets both children to None and stops. Otherwise, it
        computes the split that minimizes the Gini impurity for each child node and executes it,
        creating two new children that are the results of the split (left for false, right for
        true).
        """

        node = self.tree[pos-1]
        node_val = node.val[0]  # the tuple (data, classes)
        curr_gini = self.gini(node_val[1])

        if curr_gini == 0:  # data is pure
            self.tree.set_child(pos, 0, None)
            self.tree.set_child(pos, 1, None)  # make this a leaf
            return None  # we're done here!
        else:  # generate possible splits
            splits = self.gen_splits(*node_val)
            # find the best one
            best = min(splits, key=lambda split: self.test_split(*node_val, split))
            if len(node.val) == 1:  # no split in the node yet
                node.val.append(best)
            else:  # overwrite a split
                node.val[1].append(best)
            # execute it
            left, right = self.execute_split(*node_val, best)

            self.tree.set_child(pos, 0, Node([left], None, None))
            self.tree.set_child(pos, 1, Node([right], None, None))

    def recursively_create_splits(self, pos):
        """For the current Node, recursively splits all of its children (generating them as it goes)
        until all of the leaves are pure, returning None."""
        if pos > len(list(self.tree)) or self.tree[pos-1] is None:  # we stop here
            return None
        else:  # create two children and recurse to split them
            self.create_split(pos)
            self.recursively_create_splits(pos * 2)  # left child
            self.recursively_create_splits(pos * 2 + 1)  # right child
            return None

    def raw_fit(self, X, y):
        """Fits a given input matrix to a given output vector using a decision tree. X should be a DataFrame
        of numerical variables with any index or column names. y should be a vector or matrix with
        the same height as X, as many columns as classes to predict, and each column should be a
        list of 0's and 1's for a given class. Does no pruning.

        """
        # start with just the entire dataset at the root, with no split
        # I'm going to represent a single node of a tree as a value [(data, classes), split]
        # where data and classes are "what's left" and split is the Split function that gives the
        # left and right children, or None if the data is pure or the splits have stopped
        self.tree = Tree(Node([(X, y)], None, None))
        self.recursively_create_splits(1)  # pretty anticlimactic
        return None

    def predict(self, X):
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
