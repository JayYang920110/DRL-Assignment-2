# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
""" 

import numpy as np
import copy
from operator import itemgetter
import math
from connect6_state import Connect6State
import random

def evaluate_position(board, r, c, color):
    """Evaluates the strength of a position based on alignment potential."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    score = 0
    size = 19
    for dr, dc in directions:
        count = 1
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
            count += 1
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
            count += 1
            rr -= dr
            cc -= dc

        if count >= 5:
            score += 10000
        elif count == 4:
            score += 5000
        elif count == 3:
            score += 1000
        elif count == 2:
            score += 100
    return score


def get_top_n_moves(board, color, n):
    """回傳前n個得分最高的位置 (r, c)，優先處理立即進攻/防守"""

    size = board.shape[0]
    opponent = 3 - color
    empty_positions = [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]

    scored = []
    attack_scores = []
    defense_scores = []

    for r, c in empty_positions:
        attack = evaluate_position(board, r, c, color)
        defense = evaluate_position(board, r, c, opponent)
        attack_scores.append(((r, c), attack))
        defense_scores.append(((r, c), defense))

    # 合併攻防分數，單獨排序（每個點可能出現兩次）
    combined_scores = attack_scores + defense_scores
    combined_scores.sort(key=lambda x: x[1], reverse=True)

    # 遇到相同位置就跳過，只保留不重複的前 n 名
    seen = set()
    top_n = []
    for pos, score in combined_scores:
        if pos not in seen:
            top_n.append(pos)
            seen.add(pos)
        if len(top_n) == n:
            break

    return top_n

def rule_based_rollout_move(board, color):
    size = board.shape[0]
    opponent_color = 3 - color
    empty_positions = [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]

    # 1. Winning move
    for r, c in empty_positions:
        board[r, c] = color
        if check_win(board, color):
            board[r, c] = 0
            return (r, c)
        board[r, c] = 0

    # 2. Block opponent's winning move
    for r, c in empty_positions:
        board[r, c] = opponent_color
        if check_win(board, opponent_color):
            board[r, c] = 0
            return (r, c)
        board[r, c] = 0

    # 3. Attack or Defense based on max score
    best_move = None
    best_score = -1
    for r, c in empty_positions:
        my_score = evaluate_position(board, r, c, color)
        opp_score = evaluate_position(board, r, c, opponent_color)
        score = max(my_score, opp_score)
        if score > best_score:
            best_score = score
            best_move = (r, c)
    if best_move:
        return best_move

    # 若沒有找到最佳落點，回傳隨機一點（避免 NoneType）
    return random.choice(empty_positions)


  


def check_win(board, size=19):
    """Checks if a player has won. Returns 1 (Black wins), 2 (White wins), or 0 (no winner)."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(size):
        for c in range(size):
            if board[r, c] != 0:
                current_color = board[r, c]
                for dr, dc in directions:
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r, prev_c] == current_color:
                        continue
                    count = 0
                    rr, cc = r, c
                    while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == current_color:
                        count += 1
                        rr += dr
                        cc += dc
                    if count >= 6:
                        return current_color
    return 0


class MCTSNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, state: Connect6State, color):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self.visits = 0
        self.total_reward = 0
        self.move = None
        self.state = state
        self.color = color
     
    def select(self, c):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
    
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c))
    
    def expand(self, n):
        board = self.state.board
        color = self.color  
        top_moves = get_top_n_moves(board, color, n)

        for r1, c1 in top_moves:
    
            first_move_state = copy.deepcopy(self.state)
            first_move_state.do_move((r1, c1))
            child_node_1 = MCTSNode(parent=self, state=first_move_state, color=color)
            self._children[(r1, c1)] = child_node_1
            
            # 第二顆棋子的候選位置
            second_moves = get_top_n_moves(first_move_state.board, color, n) 
            for r2, c2 in second_moves:
                if (r2, c2) == (r1, c1):
                    continue  

                final_state = copy.deepcopy(first_move_state)
                final_state.do_move((r2, c2))

                child_node_2 = MCTSNode(parent=child_node_1, state=final_state, color=color)
                child_node_1._children[(r2,c2)] = child_node_2

    def update(self, leaf_value):
        self.total_reward += leaf_value
        self.visits += 1

    def update_recursive(self, leaf_value, from_player):

        if self._parent:
            self._parent.update_recursive(leaf_value, from_player)

        # 如果這層節點的顏色 != from_player，就翻號
        value = leaf_value if self.color == from_player else -leaf_value
        self.update(value)

    def get_value(self, c):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """

        if self.visits == 0:
            uct_score = float("inf")
        else:
            avg_reward = self.total_reward / self.visits
            exploration = c * math.sqrt(math.log(self._parent.visits) / self.visits)
            uct_score = avg_reward + exploration
        return uct_score

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """简单的实现 Monte Carlo Tree Search."""

    def __init__(self, c=1.41, n_playout=1000, n_leaf=10):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = MCTSNode(None, None, None)
        self.c = c
        self.n_leaf = n_leaf
        self._n_playout = n_playout
        self.i = 0

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        node.state = state
        node.color = state.turn
        root_color = node.color
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self.c)
            state.do_move(action)
 
        # Check for end of game
        end, winner = state.game_end()
        if end:
            if winner == -1:
                leaf_value = 0
            else:
                leaf_value = 1 if winner == root_color else -1
            node.update_recursive(leaf_value, root_color)
            return 
        if not end:
            if node.visits > 0 or node == self._root:
                node.expand(n=self.n_leaf)
                action1, child = random.choice(list(node._children.items()))
                action2, child2 = random.choice(list(child._children.items()))
                try:
                    state.do_move(action1)
                    state.do_move(action2)
                except Exception as e:
                    print(f"Invalid move during expansion: {e}", flush=True)
                    return
                leaf_value = self._evaluate_rollout(state, root_color)
                child2.update_recursive(leaf_value, root_color)

            else:
                leaf_value = self._evaluate_rollout(state, root_color)
                node.update_recursive(leaf_value, root_color)
 


    def _evaluate_rollout(self, state, root_color, limit=1000):
        def rollout_policy_fn(state):
            moves = []
         
            for _ in range(2):  # Connect6 每回合落兩顆
                move = rule_based_rollout_move(state.board, state.turn)
            
                moves.append((move, 1.0))
                state.do_move(move)
            # return moves

        
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                break
            rollout_policy_fn(state)
            # for move, _ in actions:
            #     state.do_move(move)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit", flush=True)
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == root_color else -1

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
    
        self._root = MCTSNode(None, None, None) 
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        best = None
        max_visits = -1
        for (r1, c1), node1 in self._root._children.items():
            for (r2, c2), node2 in node1._children.items():
                if node2.visits > max_visits:
                    best = ((r1, c1), (r2, c2))
                    max_visits = node2.visits

        return best  
        # for (r1, c1), node1 in self._root._children.items():
        
        #     if node1.visits > max_visits:
        #         best = (r1, c1)
        #         max_visits = node1.visits
        # return best



    def __str__(self):
        return "MCTS"


# def evaluate_position(board, r, c, color):
#     """
#     評估 (r, c) 作為 color 落子的進攻 + 防守價值
#     """
#     if board[r, c] != 0:
#         return 0

#     size = board.shape[0]
#     directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
#     opponent = 3 - color
#     score = 0

#     # 定義進攻型 pattern 分數
#     attack_patterns = {
#         f'0{color}{color}{color}{color}{color}0': 10000,
#         f'0{color}{color}{color}{color}0': 5000,
#         f'0{color}{color}{color}{color}{color}{opponent}': 4000,
#         f'{opponent}{color}{color}{color}{color}{color}0': 4000,
#         f'{opponent}{color}{color}{color}{color}0': 2000,
#         f'0{color}{color}{color}{color}{opponent}': 2000,
#         f'0{color}{color}{color}0': 1000,
#         f'{opponent}{color}{color}{color}0': 300,
#         f'0{color}{color}0': 100,
#     }

#     # 定義防守型（對手落子） pattern 分數
#     defense_patterns = {
#         f'0{opponent}{opponent}{opponent}{opponent}{opponent}0': 10000,
#         f'{color}{opponent}{opponent}{opponent}{opponent}0': 10000,
#         f'0{opponent}{opponent}{opponent}{opponent}{color}': 10000,
#     }

#     # ---------- 進攻端掃描 ----------
#     board[r, c] = color
#     for dr, dc in directions:
#         line = []
#         for i in range(-5, 6):
#             rr, cc = r + dr * i, c + dc * i
#             if 0 <= rr < size and 0 <= cc < size:
#                 line.append(str(board[rr, cc]))
#             else:
#                 line.append('-')  # 邊界當成阻擋

#         line_str = ''.join(line)
#         max_pattern_len = max(len(p) for p in attack_patterns.keys())

#         for i in range(len(line_str) - max_pattern_len + 1):
#             segment = line_str[i:i+6]
#             for pattern, p_score in attack_patterns.items():
#                 if segment == pattern:
#                     score += p_score
#     board[r, c] = 0

#     # ---------- 防守端掃描（對手落子） ----------
#     board[r, c] = opponent
#     for dr, dc in directions:
#         line = []
#         for i in range(-5, 6):
#             rr, cc = r + dr * i, c + dc * i
#             if 0 <= rr < size and 0 <= cc < size:
#                 line.append(str(board[rr, cc]))
#             else:
#                 line.append('-')

#         line_str = ''.join(line)
#         max_pattern_len_d = max(len(p) for p in defense_patterns.keys())

#         for i in range(len(line_str) - max_pattern_len_d + 1):
#             segment = line_str[i:i+6]
#             for pattern, p_score in defense_patterns.items():
#                 if segment == pattern:
#                     score += p_score
#     board[r, c] = 0

#     return score