from Caro_pybind import Caro, Point
from typing import Callable
from Model_pybind import SmallNet as SmallNetPybind

class MCTS_AI:
    def __init__(self, _player: int, _min_visits: int, _n_sim: int, _board: Caro, _mode: str,
                 _ai_moves_range: int = 1, _eval: Callable = None,
                 _prior_strength: int = 1, _random_threshold: int = 8):
        pass

    def initialize_model(self, model: SmallNetPybind):
        pass

    def get_move(self, prev_move: Point) -> Point:
        pass

    def play_move(self, move: Point) -> None:
        pass

    def switch_player(self) -> None:
        pass

    def get_uct_temperature(self) -> float:
        pass

    def set_uct_temperature(self, t: float) -> None:
        pass

    def get_play_temperature(self) -> float:
        pass

    def set_play_temperature(self, t: float) -> None:
        pass

    def get_rollout_weight(self) -> float:
        pass

    def set_rollout_weight(self, w: float) -> None:
        pass

    def get_prior_strength(self) -> int:
        pass

    def set_prior_strength(self, p: int) -> None:
        pass

    def enable_random_transform(self) -> None:
        pass

    def disable_random_transform(self) -> None:
        pass

    def get_tree_depth(self) -> int:
        pass

    def get_current_node_children_count(self) -> int:
        pass

    def get_current_node_child_move(self, index: int) -> Point:
        pass

    def get_current_node_child_average_reward(self, index: int) -> float:
        pass

    def average_child_count(self) -> float:
        pass

    def get_player(self) -> int:
        pass

    def predicted_reward(self) -> float:
        pass