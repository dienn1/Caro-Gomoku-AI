from Caro_pybind import Caro, Point
from typing import Callable
import math
import random
from copy import copy


CHAR_P = {-1: "O", 0: ".", 1: "X"}
C = math.sqrt(2)


class TreeNode:
    def __init__(self, move: Point, player: int, parent=None, turn_count: int = 0):
        self.move = move
        self.player = player
        self.parent: TreeNode = parent
        self.children: list[TreeNode] = list()

        self.visit_count = 0
        self.total_reward = 0
        self.prior_eval = 0
        self.noise = 0

        self.turn_count = 0
        if turn_count > 0:
            self.turn_count = turn_count
        elif self.parent is not None:
            self.turn_count = self.parent.turn_count + 1

    def __str__(self):
        return "TreeNode: " + str(self.move) + " player: " + CHAR_P[self.player]

    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.player * self.total_reward / self.visit_count

    def exploration_value(self, use_prior: bool = False) -> float:
        if self.visit_count + int(use_prior) == 0:
            return 9999
        return C * math.sqrt(math.log(self.parent.visit_count) / (self.visit_count + int(use_prior)))

    def uct(self) -> float:
        return self.average_reward() + self.exploration_value()


class MCTS:
    def __init__(self, _player: int, _min_visits: int, _n_sim: int, _board: Caro, _mode: str,
                 _ai_moves_range: int = 1, _eval: Callable = None,
                 _prior_strength: int = 1, _random_threshold: int = 8):
        self.player = _player
        self.min_visits = _min_visits
        self.n_sim = _n_sim
        self.board: Caro = copy(_board)

        self.current_node: TreeNode = None
        self.AI_moves_range = _ai_moves_range
        self.mode = _mode
        self.random_threshold = _random_threshold

        self.use_prior = False
        self.evaluate_prior = _eval
        self.prior_strength = 1
        if self.evaluate_prior is not None:
            self.use_prior = True
            if _prior_strength >= 0:
                self.prior_strength = _prior_strength

        self.child_count = 0
        self.expanded_nodes_count = 0
        self.current_depth = 0
        self.current_max_depth = 0

    def expand_node(self, node: TreeNode) -> None:
        self.expanded_nodes_count += 1
        depth = node.turn_count + 1
        if depth > self.current_max_depth:
            self.current_max_depth = depth

        moves = self.board.get_AI_moves()
        board_temp = self.board.get_board()
        for p in moves:
            child = TreeNode(p, -node.player, node, depth)
            node.children.append(child)
            # Evaluate prior for each node if a prior is given
            if self.use_prior:
                board_temp[p(0)][p(1)] = child.player
                child.prior_eval = float(self.evaluate_prior(board_temp, self.board.get_dim()))
                board_temp[p(0)][p(1)] = 0

        self.child_count += len(node.children)

    def posterior_eval(self, node: TreeNode) -> float:
        if self.prior_strength == 0 and node.visit_count == 0:
            return 0
        return node.player * (node.prior_eval * self.prior_strength + node.total_reward) / (self.prior_strength + node.visit_count)

    def evaluate_node(self, node: TreeNode) -> float:
        if self.use_prior:
            return self.posterior_eval(node) + node.exploration_value(self.use_prior)
        return node.uct()

    def mcts(self, node: TreeNode) -> float:
        if node.visit_count >= self.min_visits:     # Matured Node
            node.visit_count += 1
            if len(node.children) == 0:             # Initialize children if empty
                self.expand_node(node)
            next_node = self.mcts_selection(node)
            self.board.play(next_node.move)
            if self.board.has_ended():              # If the game ends, accumulate reward for next and current node
                next_node.visit_count += 1
                result = self.board.get_state()     # backprop result
                next_node.total_reward += result
                node.total_reward += result
                self.board.undo()
                return result
            result = self.mcts(next_node)           # recursion call mcts on node next if game doesn't end
            self.board.undo()
            # backprop result back up
            node.total_reward += result
            return result
        else:   # not enough maturity
            node.visit_count += 1
            if self.use_prior:
                result = node.prior_eval
            else:
                result = self.simulate()
            if node.visit_count >= self.min_visits:  # if matured
                self.expand_node(node)
            # backprop result back up
            node.total_reward += result
            return result

    def simulate(self) -> int:
        current_turn = self.board.get_turn_count()
        self.board.simulate()
        end_turn = self.board.get_turn_count()
        final_state = self.board.get_state()
        for i in range(end_turn - current_turn):
            self.board.undo()
        return final_state

    def mcts_selection(self, node: TreeNode) -> TreeNode:
        current = node.children[0]
        current_eval = self.evaluate_node(current)
        for child in node.children:
            child_eval = self.evaluate_node(child)
            if current_eval < child_eval:
                current = child
                current_eval = child_eval
        return current

    def posterior_selection(self, node: TreeNode) -> TreeNode:
        current = node.children[0]
        for child in node.children:
            if self.posterior_eval(current) < self.posterior_eval(child):
                current = child
        return current

    @staticmethod
    def reward_selection(node: TreeNode) -> TreeNode:
        current = node.children[0]
        for child in node.children:
            if current.average_reward() < child.average_reward():
                current = child
        return current

    @staticmethod
    def random_selection(node: TreeNode) -> TreeNode:
        rand_index = random.randrange(len(node.children))
        return node.children[rand_index]

    @staticmethod
    def visit_selection(node: TreeNode) -> TreeNode:
        current = node.children[0]
        for child in node.children:
            if current.visit_count < child.visit_count:
                current = child
        return current

    @staticmethod
    def weighted_visit_selection(node: TreeNode) -> TreeNode:
        weights = list(child.visit_count for child in node.children)
        return random.choices(node.children, weights=weights)[0]

    def get_move(self, prev_move: Point) -> Point:
        if prev_move != Point(-1, -1):
            self.board.play(prev_move)
        # AI first move, current_node will be None
        if self.current_node is None:
            self.current_node = TreeNode(prev_move, -self.player, None, self.board.get_turn_count())
            self.expand_node(self.current_node)
        else:   # update current_node to be its child with prev_move
            for child in self.current_node.children:
                if child.move == prev_move:
                    self.current_node = child
                    break
        # Initialize child nodes if empty
        if len(self.current_node.children) == 0:
            self.expand_node(self.current_node)

        self.current_depth = self.current_node.turn_count
        # MCTS for self.n_sim iterations
        for n in range(self.n_sim):
            self.mcts(self.current_node)

        if self.mode == "random":
            self.current_node = self.random_selection(self.current_node)
        elif self.mode == "greedy_post":
            self.current_node = self.posterior_selection(self.current_node)
        elif self.mode == "greedy":
            self.current_node = self.reward_selection(self.current_node)
        elif self.mode == "greedy_visit":
            self.current_node = self.visit_selection(self.current_node)
        elif self.mode == "weighted_visit":
            self.current_node = self.weighted_visit_selection(self.current_node)
        elif self.mode == "alpha_zero":
            if self.board.get_turn_count() <= self.random_threshold:
                self.current_node = self.weighted_visit_selection(self.current_node)
            else:
                self.current_node = self.visit_selection(self.current_node)
        else:
            print("INVALID MODE")
            raise ValueError("INVALID MODE REEEEEEEEEEEE")
        self.board.play(self.current_node.move)
        # Initialize child nodes if empty
        if len(self.current_node.children) == 0:
            self.expand_node(self.current_node)
        self.current_node.parent = None     # Garbage collector the rest of the trees
        return self.current_node.move

    def get_tree_depth(self) -> int:
        return self.current_max_depth - self.current_depth

    def average_child_count(self) -> float:
        if self.expanded_nodes_count == 0:
            return 0
        return self.child_count/self.expanded_nodes_count

    def get_player(self) -> int:
        return self.player

    def predicted_reward(self) -> float:
        if self.mode == "greedy_post":
            return self.player * self.posterior_eval(self.current_node)
        return self.player * self.current_node.average_reward()
