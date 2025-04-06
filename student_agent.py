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
from td_mcts import TD_MCTS, PlayerNode
from approximator import NTupleApproximator

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


approximator = None
already_printed_2048 = False
already_printed_4096 = False
already_printed_8192 = False
already_printed_16384 = False
def get_action(state, score):

    global approximator, already_printed_4096, already_printed_2048, already_printed_8192, already_printed_16384
    
    if approximator is None:
        patterns = [
                    [(0,0), (0,1), (1,0), (1,1),(2,0), (2,1)],
                    [(0,0), (0,1), (1,1), (1,2), (1,3), (2,2)],
                    [(0,0), (1,0), (2,0), (2,1), (3,0), (3,1)],
                    [(0,1), (1,1), (2,1), (2,2), (3,1), (3,2)],
                    [(0,0), (0,1), (0,2), (1,1), (2,1), (2,2)],
                    [(0,0), (0,1), (1,1), (2,1), (3,1), (3,2)],
                    [(0,0), (0,1), (1,1), (2,1), (2,0), (3,1)],
                    [(0,0), (1,0), (0,1), (0,2), (1,2), (2,2)]]
        approximator = NTupleApproximator(board_size=4, patterns=patterns)

        try:
            approximator.load("ntuple_weights.pkl")
            print("load successfully!")
        except Exception as e:
            print("[ERROR] 加載權重失敗:", e)


    env = Game2048Env()
    env.board = np.array(state, dtype=int) 
    if np.all(env.board == 0):
        already_printed_2048 = False
        already_printed_4096 = False
        already_printed_8192 = False
        already_printed_16384 = False
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
    elif not already_printed_8192:
        max_tile = np.max(env.board)
        if max_tile == 8192:
            print(f"[INFO] Max tile reached: {max_tile}")
            already_printed_8192 = True  
    elif not already_printed_16384:
        max_tile = np.max(env.board)
        if max_tile == 16384:
            print(f"[INFO] Max tile reached: {max_tile}")
            already_printed_16384 = True 

    env.score = score  
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 0  

    root = PlayerNode(state=env.board.copy(), score=env.score, env=env)
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=0.0, rollout_depth=0, gamma=0.99)

    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_action, distribution = td_mcts.best_action_distribution(root)
    return best_action
    # best_value = float('-inf')
    # best_action = None
    # for a in legal_moves:
    #     sim_env = copy.deepcopy(env)
    #     # sim_env.step(a, spawn_tile=False)
    #     _, reward, _, _ = sim_env.step(a, spawn_tile=False)
    #     after_state = sim_env.board.copy()
    #     # value = approximator.value(after_state)
    #     value = reward + approximator.value(after_state)

    #     if value > best_value:
    #         best_value = value
    #         best_action = a

    # return best_action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.







