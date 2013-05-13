import numpy as np
from itertools import *


class Node:
    def __init__(self, bounds):
        self.bounds = bounds
        self.leaf = True
        self.children = []

    def descend(self):
        """ Generate children of this node.
        """
        for ncube in split(self.bounds):
            self.children += [Node(ncube)]
        return self.children

    def get_bounds(self):
        """ Return boundary as a 2d array containing
        mininum and maximum in each dimension
        """
        return np.array([np.amin(self.bounds, axis=0),
                        np.amax(self.bounds, axis=0)], dtype=float)


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

def split(ncube):
    """ Split an n-cube into 2^n equally sized
    child n-cubes.
    """
    # Shrink the cube
    bottom_left = ncube / 2.
    # Find the length of a side
    c = bottom_left[1] - bottom_left[0]
    offset = np.sqrt(c.dot(c))
    # Find how much shrinking it has moved it
    d = ncube[0] - bottom_left[0]
    # Move it back
    bottom_left += d
    # Get moves in all directions away from bottom corner
    offsets = product([0, offset], repeat=len(c))
    # Make 2^n shifted copies
    children = [bottom_left.copy() + np.array(x) for x in offsets]
    return np.array(children)