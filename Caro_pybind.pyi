class Point:
    def __init__(self, x: int, y: int):
        pass

    def __str__(self):
        pass

    def __call__(self, i: int) -> int:
        pass


class Caro:
    def __init__(self, _dim: int = 19, count: int = 5, _ai_moves_range: int = 1):
        pass

    def __str__(self):
        pass

    def simulate(self, n_turns:int = -1) -> None:
        pass

    def disable_print(self) -> None:
        pass

    def enable_print(self) -> None:
        pass

    def get_moves(self) -> set[Point]:
        pass

    def get_prev_move(self) -> Point:
        pass

    def get_state(self) -> int:
        pass

    def has_ended(self) -> bool:
        pass

    def current_player(self) -> int:
        pass

    def get_turn_count(self) -> int:
        pass

    def get_dim(self) -> int:
        pass

    def get_board(self) -> list[list[int]]:
        pass

    def get_AI_moves_range(self) -> int:
        pass

    def set_AI_moves_range(self, moves_range:int) -> None:
        pass

    def play(self, pos: Point) -> bool:
        pass

    def undo(self) -> None:
        pass

    def get_random_move(self) -> Point:
        pass

    def get_AI_moves(self) -> set[Point]:
        pass

    def get_moves_history(self) -> list[Point]:
        pass
