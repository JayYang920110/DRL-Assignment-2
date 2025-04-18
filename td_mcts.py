import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class PlayerNode:
    def __init__(self, state, score, parent=None, action=None, env=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}  # action -> ChanceNode
        self.visits = 0
        self.total_reward = 0.0
        self.legal_actions = {}  # action -> (afterstate, after_score)
        if env is not None:
            for a in range(4):
                sim = copy.deepcopy(env)
                sim.board = state.copy()
                sim.score = score
                board, new_score, done, _ = sim.step(a, spawn_tile=False)
                if not np.array_equal(state, board):
                    self.legal_actions[a] = (board, new_score)
        
        self.untried_actions = list(self.legal_actions.keys())

    def fully_expanded(self):
        if not self.legal_actions:  # 沒有合法動作 → 不需要展開
            return False
        return all(action in self.children for action in self.legal_actions)
    def is_leaf(self):
        # 如果還有未展開的動作 → 是 leaf
        return not self.fully_expanded()

class ChanceNode:
    def __init__(self, state, score, parent, action):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action  # The action taken to reach this node (by player)
        self.children = {}  # (pos, val) -> PlayerNode
        self.visits = 0
        self.total_reward = 0.0
        self.expanded = False  

    def is_leaf(self):
        return not self.expanded
    
    def fully_expanded(self, empty_tiles):
        return len(self.children) == len(empty_tiles) * 2  # For 2 and 4

#value
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        
        self.approximator = approximator
        self.min_value_seen = float('inf')
        self.max_value_seen = float('-inf')

    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env
    
    def evaluate_best_afterstate_value(self, sim_env, approximator):
        temp_node = PlayerNode(sim_env.board.copy(), sim_env.score, env=sim_env)
        if not temp_node.legal_actions:
            return 0
        
        max_value = float('-inf')
        for a, (board, new_score) in temp_node.legal_actions.items():
            reward = new_score - sim_env.score
            v = reward + approximator.value(board)
            max_value = max(max_value, v)
        return max_value
    
    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        best_uct_score = -float("inf")
        best_child = None
        best_action = None
        for action, child in node.children.items():
            if child.visits == 0:
                uct_score = self.approximator.value(child.state)
            else:
                avg_reward = child.total_reward / child.visits
                # print(avg_reward)
                exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
                uct_score = avg_reward + exploration
                # print(f"reward:{avg_reward}, exploration:{math.sqrt(math.log(node.visits) / child.visits)}, ratio:{avg_reward/exploration}")
            if uct_score > best_uct_score:
                best_child = child
                best_action = action
                best_uct_score = uct_score
        # if best_child is None:
        #     print("Error")
        return best_action, best_child
    
    def select(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        rewards = []
        while not node.is_leaf():

            if isinstance(node, PlayerNode):
                # print(f"[Debug] node.is_leaf(): {node.is_leaf()}")
                # print(f"legal_actions: {list(node.legal_actions.keys())}")
                # print(f"children: {list(node.children.keys())}")
                action, _ = self.select_child(node)
                prev_score = sim_env.score
                _, new_score, done, _ = sim_env.step(action, spawn_tile=False)
                reward = new_score - prev_score
                rewards.append(reward)


                if action not in node.children:
                    node.children[action] = ChanceNode(sim_env.board.copy(), new_score, parent=node, action=action)
                node = node.children[action]

            elif isinstance(node, ChanceNode):

                keys = list(node.children.keys())  # key: (pos, val)
                weights = [0.9 if val == 2 else 0.1 for (_, val) in keys]
                sampled_key = random.choices(keys, weights=weights, k=1)[0]

                node = node.children[sampled_key]
                sim_env = self.create_env_from_state(node.state, node.score)
        return node, sim_env, rewards
    
    def expand(self, node, sim_env):
        if sim_env.is_game_over():
            return node, sim_env

        if isinstance(node, PlayerNode) and not node.children:
            for action, (board, new_score) in node.legal_actions.items():

                chance_node = ChanceNode(board.copy(), new_score, parent=node, action=action)
                node.children[action] = chance_node
  

        elif isinstance(node, ChanceNode) and not node.expanded:
            self.expand_chance_node(node)

    def rollout(self, node, sim_env):

        if isinstance(node, PlayerNode):
            value = self.evaluate_best_afterstate_value(sim_env, self.approximator)

        elif isinstance(node, ChanceNode):
            value = self.approximator.value(node.state)

        else:
            value = 0  # fallback

        # Normalization（如果有探索常數 c）
        if self.c != 0:
            self.min_value_seen = min(self.min_value_seen, value)
            self.max_value_seen = max(self.max_value_seen, value)
            if self.max_value_seen == self.min_value_seen:
                normalized_return = 0.0
            else:
                normalized_return = 2 * (value - self.min_value_seen) / (self.max_value_seen - self.min_value_seen) - 1
        else:
            normalized_return = value

        return normalized_return

    # def backpropagate(self, node, value, rewards):
    #     G = value
    #     node.visits += 1
    #     node.total_reward += G
    #     node = node.parent
        
    #     for r in reversed(rewards):
    #         G = r + self.gamma * G
    #         node.visits += 1
    #         node.total_reward += G
    #         node = node.parent
    def backpropagate(self, node, value, rewards):
        G = value
        reward_idx = len(rewards) - 1
        if isinstance(node, ChanceNode):
            node.total_reward += G
            node.visits += 1
            node = node.parent
        elif isinstance(node, PlayerNode):
            node.visits += 1
            node = node.parent
            if node is not None and isinstance(node, ChanceNode):
                node.total_reward += G
                node.visits += 1
                node = node.parent
                
        while node is not None:
            node.visits += 1
            if isinstance(node, ChanceNode):
                reward = rewards[reward_idx]
                reward_idx -= 1
                G = G + reward
                node.total_reward += G
            node = node.parent

    def expand_chance_node(self, node):
        empty_tiles = list(zip(*np.where(node.state == 0)))

        for pos in empty_tiles:
            for val in [2, 4]:
                new_state = node.state.copy()
                new_state[pos] = val
                key = (pos, val)
                if key not in node.children:
                    child = PlayerNode(new_state, node.score, parent=node, action=key, env=self.env)
                    node.children[key] = child

        node.expanded = True
        
    def run_simulation(self, root):

        # --- Selection ---
        node, sim_env, rewards = self.select(root)

        # --- Expansion ---
        self.expand(node, sim_env)

        # --- Rollout ---
        value = self.rollout(node, sim_env)

        # --- Backpropagation ---
        self.backpropagate(node, value, rewards)

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

