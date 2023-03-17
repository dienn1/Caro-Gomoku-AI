from Caro_pybind import Caro, Point
from typing import Callable

class MCTS_AI:
    def __init__(self, _player: int, _min_visits: int, _n_sim: int, _board: Caro, _mode: str,
                 _ai_moves_range: int = 1, _eval: Callable = None,
                 _prior_strength: int = 1, _random_threshold: int = 8):
        pass

    def get_move(self, prev_move: Point) -> Point:
        pass

    def get_tree_depth(self) -> int:
        pass

    def average_child_count(self) -> float:
        pass

    def get_player(self) -> int:
        pass

    def predicted_reward(self) -> float:
        pass