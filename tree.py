"""This file provides a base Tree class for use with the DecisionTree class. It is simply a root
node (which has children, which have children, etc.), with some useful functions for doing things
you might want to do with a tree structure.

Team: Jocelyn, Kevin, Nicholas
Author: Nicholas
"""


from itertools import chain


class Tree:
    """Represents a binary tree."""
    def __init__(self, root_node):
        """Takes in a root node with children, who have children, and so on.  Traverses the node to build
        the tree, so make sure that you don't have an infinite loop!
        """
        self.root = root_node
        self.__calculate_node_list()

    def __calculate_node_list(self):
        """Internal method for reconstructing the nodes list when a change happens."""
        self.nodes = [self.root]
        children = self.root.children()
        self.nodes += list(children)

        while not all([c is None for c in children]):  # still values left
            new_children = []
            for child in children:
                if child is None:
                    new_children += [None, None]
                else:
                    new_children += list(child.all_children())
            children = new_children
            self.nodes += children

    def __str__(self):
        """Prints out a basic string representation."""
        return str(self.root)

    def __repr__(self):
        """Basic repr. Skips the recursive Node repr, as that could be EXTREMELY long."""
        return "Tree({})".format(repr(self.root.val))

    def breadth_first_iterable(self):
        """Returns an iterator over the tree in breadth_first order. There might a a lot of None
        values, because the entire tree is filled until the deepest node."""
        return iter(self.nodes)

    def __iter__(self):
        """Trying to iterate over this class should return the nodes in breadth_first order. Lots of
        Nones galore!"""
        return iter(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]

    def __depth_first_next(self, i):
        """Internal method for depth_first_iterable. Takes a 1-index to the nodes list and goes
        through the following recursive step: if the node has children, return the left child's
        result and the right child's result combined, with the current node at the beginning. If the
        node is a leaf, return itself."""
        node = self.nodes[i-1]
        l = self.__depth_first_next(i * 2) if node.left is not None else []
        r = self.__depth_first_next(i * 2 + 1) if node.right is not None else []
        return [i, *l, *r]

    def depth_first_iterable(self):
        """Returns an iterator over the tree in depth_first order. For example, a complete 3-deep binary
        tree numbered breadth-first 1, 2, 3, 4, 5, 6, 7 (1 -> 2 -> 4 is the left edge, for
        clarity) would return an iterator over the nodes in the order 1, 2, 4, 5, 3, 6, 7.

        """
        curr = 1   # "walk" down the tree using 1-indexed indices of the breadth-first children
        depth_first_nodes = self.__depth_first_next(curr)
        return [self.nodes[i - 1] for i in depth_first_nodes]  # subtract 1 to get the actual index

    def set_child(self, position, direction, child):
        """Sets a child at a given point on the tree (using breadth-first indexing). Child is the Node to
        add (or None to remove a child), position is the index of the parent, and direction is 0 for
        left side or 1 for right side. Raises ValueError if you attempt to delete a parent.

        As always, breadth-first indexing means 1-indexed as well.

        """
        parent = self.nodes[position-1]
        if child is None and parent.all_children()[direction] is not None:
            # can't delete a parent!
            raise ValueError("Can't delete a parent! Think of the children!")
        else:
            if direction == 0:
                parent.left = child
            else:
                parent.right = child
        self.__calculate_node_list()

    def index(self, node_val):
        """Gets the 1-indexed index of a given node value, the topmost furthest to the left if there
        are multiple. Ignores the children. Returns None if nothing was found."""
        for i, node in enumerate(self.nodes):
            if node.val == node_val:
                return i + 1
        return None

    def search_all(self, node_val):
        """Returns a list of all node indexes (1-indexed breadth-first) that have the given node value,
        ordered top-down and left-to-right.

        """
        ans = []
        for i, node in enumerate(self.nodes):
            if node.val == node_val:
                ans.append(i + 1)
        return ans

    def get_rows(self):
        """Gets a list of lists, where each list is the next-deepest row. Does not fill gaps with
        None values."""
        rows = []
        rows.append([self.root])
        children = self.root.children()
        while children:  # still values left
            rows.append(list(children))
            children = list(chain.from_iterable([child.children() for child in children]))
        return rows

    def get_row(self, row_depth):
        """Gets a list of all the nodes at a given height, where 0 is the root node. Does not fill
        gaps with None values. Wraps get_rows"""
        return self.get_rows()[row_depth]

    def depth(self):
        """Returns the length of the longest walk down the tree."""
        return len(self.get_rows())
