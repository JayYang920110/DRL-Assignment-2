# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
    
        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action, spawn_tile=True):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved and spawn_tile:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}


    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)
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


    # def value(self, board):
    #     # TODO: Estimate the board value: sum the evaluations from all patterns.
    #     # total = 0.0
    #     # for i, sys in enumerate(self.patterns):
    #     #     feature = self.get_feature(board, sys)
    #     #     total += self.weights[i][feature]
    #     # total = total / len(self.patterns)
    #     # return total
    #     total = 0.0
    #     count = 0
    #     for i, sym_group in enumerate(self.symmetry_patterns):
    #         for coords in sym_group:
    #             feature = self.get_feature(board, coords)
    #             total += self.weights[i][feature]
    #             count += 1
    #     return total / count if count > 0 else 0.0
    def value(self, board):
        total = 0.0
        for p in range(len(self.patterns)):
            pattern_sum = 0.0
            for sym in self.symmetry_patterns[p]:
                index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)
                pattern_sum += self.weights[p][index]
            total += pattern_sum / 8.0  # 對8個對稱平均
        return total
    # def update(self, board, delta, alpha):
    #     # TODO: Update weights based on the TD error.
    #     # num_pattern = len(self.patterns)
    #     # normalized_alpha = alpha / num_pattern
    #     # for i, sys in enumerate(self.patterns):
    #     #         feature = self.get_feature(board, sys)
    #     #         self.weights[i][feature] += delta * alpha
    #     for i, sym_group in enumerate(self.symmetry_patterns):  # 每個 pattern 的對稱群組
    #         for coords in sym_group:  # 這組 pattern 的所有對稱版本
    #             feature = self.get_feature(board, coords)
    #             self.weights[i][feature] += (alpha * delta) / len(sym_group) 
    #             # self.weights[i][feature] += (alpha * delta) 
    def update(self, board, delta, alpha):
        for p in range(len(self.patterns)):  # 遍歷所有 pattern
            for sym in self.symmetry_patterns[p]:  # 遍歷該 pattern 的所有對稱版本
                index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)  # 將棋盤上的 tile 值轉換為索引
                self.weights[p][index] += alpha * delta  # 更新對應特徵的權重
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    
    def load(self, filename):
        with open(filename, 'rb') as f:
            simple_weights = pickle.load(f)
        # 包成 defaultdict(float)
        self.weights = [defaultdict(float, w) for w in simple_weights]

approximator = None
already_printed_2048 = False
already_printed_4096 = False
def get_action(state, score):
    global approximator, already_printed_4096, already_printed_2048
    if approximator is None:
        # patterns = []
        # for row in range(4):
        #     pattern = [(row, col) for col in range(4)]
        #     patterns.append(pattern)

        # for col in range(4):
        #     pattern = [(row, col) for row in range(4)]
        #     patterns.append(pattern)

        # for row in range(3): 
        #     for col in range(3):  
        #         pattern = [
        #             (row, col),
        #             (row, col+1),
        #             (row+1, col),
        #             (row+1, col+1)
        #         ]
        #         patterns.append(pattern)
        patterns = [[(0,0), (0,1), (1,0), (1,1),(2,0), (2,1)],
                [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)],
                [(0,0), (1,0), (2,0), (2,1), (3,0), (3,1)],
                [(0,1), (1,1), (2,1), (2,2), (3,1), (3,2)]]

        approximator = NTupleApproximator(board_size=4, patterns=patterns)
        approximator.load("ntuple_weights.pkl")

    env = Game2048Env()
    env.board = np.array(state, dtype=int) 
    if np.all(env.board == 0):
        already_printed_2048 = False
        already_printed_4096 = False

    if not already_printed_2048:
        max_tile = np.max(env.board)
        if max_tile == 2048:
            print(f"[INFO] Max tile reached: {max_tile}")
            already_printed_2048 = True  
    elif not already_printed_4096:
        max_tile = np.max(env.board)
        if max_tile == 4096:
            print(f"[INFO] Max tile reached: {max_tile}")
            already_printed_4096 = True  
    
    env.score = score  
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 0  # 如果沒合法動作，回傳預設值

    # 模擬每個動作的結果，選擇估價值最大的
    best_value = float('-inf')
    best_action = None
    for a in legal_moves:
        sim_env = copy.deepcopy(env)
        # sim_env.step(a, spawn_tile=False)
        _, reward, _, _ = sim_env.step(a, spawn_tile=False)
        after_state = sim_env.board.copy()
        # value = approximator.value(after_state)
        value = reward + approximator.value(after_state)

        if value > best_value:
            best_value = value
            best_action = a

    return best_action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.







