import numpy as np


class Node:
    def __init__(self, bounds):
        self.bounds = bounds
        self.leaf = True
        self.children = []

    def descend(self):
        """ Generate children of this node.
        """
        for square in split_square(self.bounds):
            self.children += [Node(square)]
        return self.children

    def get_bounds(self):
        """ Return boundary as a 2d array containing
        mininum and maximum in each dimension
        """
        return np.array([np.amin(self.bounds, axis=0),
                        np.amax(self.bounds, axis=0)])


class Tree:
    def __init__(self, bounds, depth):
        self.root = Node(bounds)
        self.depth = depth
        self.levels = [[self.root]]

    def generate(self):
        """ Build a tree to a specified depth and
        record the nodes at each level.
        """
        for i in range(self.depth):
            nodes = []
            for node in self.levels[i]:
                nodes += node.descend()
            self.levels += [nodes]

    def get_bounds(self):
        """ Get the bounds of all nodes as a flat list root -> children..
        """
        return [item.get_bounds() for sublist in self.levels for item in sublist]


def split_square(square):
    """ Subdivide a square into 4 equally sized smaller
    squares and return an array of arrays of vertices.
    """
    v1_a = (square[3] + square[0]) / 2.
    v2_a = (square[0] + square[1]) / 2.
    v3_a = (square[1] + square[2]) / 2.
    v4_a = (square[2] + square[3]) / 2.
    v5 = np.sum(square, axis=0) / 4.

    return np.array([np.array([square[0], v2_a, v5, v1_a]),
                    np.array([v2_a, square[1], v3_a, v5]),
                    np.array([v1_a, v5, v4_a, square[3]]),
                    np.array([v5, v3_a, square[2], v4_a])])
