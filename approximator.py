import pickle
import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from collections import namedtuple

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rot90(pattern):
    return [(y, 3 - x) for (x, y) in pattern]

def rot180(pattern):
    return [(3 - x, 3 - y) for (x, y) in pattern]

def rot270(pattern):
    return [(3 - y, x) for (x, y) in pattern]

def flip_horizontal(pattern):
    return [(x, 3 - y) for (x, y) in pattern]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
    #     # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    # def canonical_feature(self, board, symmetries):
    #     return min(self.get_feature(board, coords) for coords in symmetries)
    
    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = [
            pattern,
            rot90(pattern),
            rot180(pattern),
            rot270(pattern),
            flip_horizontal(pattern),
            rot90(flip_horizontal(pattern)),
            rot180(flip_horizontal(pattern)),
            rot270(flip_horizontal(pattern))
        ]
        return syms
    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for (x, y) in coords)

    def value(self, board):
        total = 0.0
        for p in range(len(self.patterns)):
            pattern_sum = 0.0
            for sym in self.symmetry_patterns[p]:
                index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)
                pattern_sum += self.weights[p][index]
            total += pattern_sum / 8.0  # 對8個對稱平均
        return total / len(self.patterns)
    
    def update(self, board, delta, alpha):
        for p in range(len(self.patterns)):  # 遍歷所有 pattern
            for sym in self.symmetry_patterns[p]:  # 遍歷該 pattern 的所有對稱版本
                index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)  # 將棋盤上的 tile 值轉換為索引
                self.weights[p][index] += alpha * delta
        return self.value(board)
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)