"""This file implements a Node, a very simple class that simply has a left and a right child (both
Nodes or None), and a value.

Team: Jocelyn, Kevin, Nicholas
Author: Nicholas
"""

from functools import total_ordering  # neat trick


@total_ordering   # now every comparison function is defined. Neato!
class Node:
    """A Node of a tree."""

    def __init__(self, val, left, right):
        """Represents a node of a binary tree.  Val is the value this Node has: can be anything but should
        probably be hashable. Left and right are child Nodes, or None if this Node currently has no
        child there."""
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        """Gives the Node's value and the values of its children."""
        return self.str_with_offset(self)

    @classmethod
    def str_with_offset(cls, node, offset=0):
        """Generates a string printout of the node, but adds a given offset to every line for
        alignment. If node is None, returns "None".
        """
        if node is None:
            return None
        ans = []  # create list of lines and join at the end
        ans.append("{}".format(str(node.val)))
        ans.append("\n{}==> {} ".format(' ' * offset, cls.str_with_offset(node.left, offset+4)))
        ans.append("\n{}==> {} ".format(' ' * offset, cls.str_with_offset(node.right, offset+4)))
        return ''.join(ans)

    def __repr__(self):
        """Gives the possible constructor."""
        return "Node({}, {}, {})".format(repr(self.val), repr(self.left), repr(self.right))

    def children(self):
        """Returns a tuple (L, R) of children, if both exist, otherwise just the one that does, or (). Note
        that this only returns a truthy value if the node has children: very useful!
        """
        if self.left is None:
            return () if self.right is None else (self.right,)
        elif self.right is None:
            return (self.left,)
        else:
            return (self.left, self.right)

    def all_children(self):
        """Returns a tuple (L, R) of children, but includes None instead of removing it."""
        return (self.left, self.right)

    def __eq__(self, other):
        """Uses the value, not the children, in comparison."""
        return self.val == other.val

    def __le__(self, other):
        """Uses the value, not the children, in comparison."""
        return self.val <= other.val
