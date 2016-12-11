"""This is a class for encoding a split at a node of a decision tree in a class. That way, we can
represent more than one kind of split as both a class method function that splits data, and as a
logical condition in a human-readable format.

Team: Jocelyn, Kevin, Nicholas
Author: Nicholas
"""


class Split:
    """Represents a decision tree split. For now, this is just a less-than-or-equal-to test with a
    given value, but it could be extended in the future."""

    def __init__(self, predictor_name, val):
        """Val is the value that the split checks for <= relation to: for example, val=3.6 would make a
        Split to test x <= 3.6 for some input x. Predictor_name is the name of the predictor to
        test: for example, in a decision tree with a predictor y_1, you could use "y_1" as the
        predictorname to check if y_1 <= 3.6.

        """
        self.predictor_name = predictor_name
        self.val = val

    def __str__(self):
        """No explanation needed!"""
        return "{{} <= {}}".format(str(self.predictor_name), str(self.val))

    def __repr__(self):
        """Also no explanation needed."""
        return "Split({}, {})".format(repr(self.predictor_name), repr(self.val))

    def split(self, x, predictor_matrix, use_iloc=False):
        """Predictor_matrix must be a DataFrame with the given predictor name in it. X is then the index row
        to use for splitting, or, if use_iloc is True, the numerical index of a row (starting at
        0). Returns the result of the split applied to that particular entry.
        """
        if use_iloc:
            entry = predictor_matrix[self.predictor_name].iloc[x]
        else:
            entry = predictor_matrix[self.predictor_name][x]

        return entry <= self.val
