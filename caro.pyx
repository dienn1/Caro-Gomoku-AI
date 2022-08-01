# from __future__ import annotations
import numpy as np
import random
import time
import cython


if cython.compiled:
    print("COMPILED VERSION AAAAA")
else:
    print("INTERPRETER")

DOWN: (cython.int, cython.int) = (1, 0)
UP: (cython.int, cython.int) = (-1, 0)
RIGHT: (cython.int, cython.int) = (0, 1)
LEFT: (cython.int, cython.int) = (0, -1)
DOWN_RIGHT: (cython.int, cython.int) = (1, 1)
DOWN_LEFT: (cython.int, cython.int) = (1, -1)
UP_RIGHT: (cython.int, cython.int) = (-1, 1)
UP_LEFT: (cython.int, cython.int) = (-1, -1)

CHAR: dict = {1: "X", -1: "O", 0: "."}
MAX_DIM: cython.int = 30

@cython.cclass
class Caro:
    COUNT: cython.int
    turn_count: cython.int
    dim: cython.int
    size: cython.int
    board: cython.int[30][30]
    game_state: cython.int
    game_ended: cython.bint
    turn: cython.int
    prev_move: (cython.int, cython.int)
    AI_moves: set

    def __init__(self, dim: cython.int = 19):
        self.COUNT = 5
        self.turn_count = 0

        self.dim = min(dim, MAX_DIM)
        self.size = self.dim * self.dim
        # self.board = np.zeros((self.dim, self.dim))
        # self.board = [[0 for _ in range(self.dim)] for i in range(self.dim)]
        self.board = [[0]*30]*30

        self.game_state = 0     # 0:Undecided, 1: X wins, -1: Y wins
        self.game_ended = False
        self.turn = 1   # 1:X  -1:O
        self.prev_move = (-1, -1)

        self.AI_moves = {(int(self.dim/2), int(self.dim/2))}

    def __str__(self) -> str:
        res: str = ""
        row_str: str
        i: cython.int
        j: cython.int
        for i in range(self.dim):
            row_str = ""
            for j in range(self.dim):
                row_str += CHAR[self.board[i][j]] + " "
            res += row_str + '\n'

        return res

    @cython.cfunc
    def in_bound(self, pos: (cython.int, cython.int)) -> cython.bint:
        return self.dim > pos[0] >= 0 and self.dim > pos[1] >= 0

    @cython.cfunc
    def is_unoccupied(self, pos: (cython.int, cython.int)) -> cython.bint:   # not using in_bound for speed
        return self.dim > pos[0] >= 0 and self.dim > pos[1] >= 0 and self.board[pos[0]][pos[1]] == 0

    def get_moves(self) -> set:
        return self.AI_moves

    @cython.ccall
    def get_state(self) -> cython.int:
        return self.game_state

    @cython.ccall
    def has_ended(self) -> cython.bint:
        return self.game_ended

    # Place a piece at pos on board
    # turn argument for debugging
    @cython.ccall
    def play(self, pos: (cython.int, cython.int)) -> cython.bint:
        if self.game_ended:
            print("GAME ALREADY ENDED")
            return False
        if not self.in_bound(pos):
            print(pos, "OUT OF BOUND")
            return False
        if self.board[pos[0]][pos[1]] == 0:
            self.board[pos[0]][pos[1]] = self.turn
            self.prev_move = pos
            self.turn_count += 1
            self._check_win()
            self._generate_AI_moves()
            self.turn = -self.turn      # switch turn
            return True
        else:
            print(pos, "already occupied")
            return False

    # FOR AI USAGE, give AI a set of more limited valid moves
    # generate moves that are at most n distance away from prev_pos
    @cython.cfunc
    def _generate_AI_moves(self, n: cython.int = 2) -> cython.void:
        pos: (cython.int, cython.int) = self.prev_move
        self.AI_moves.discard(pos)
        # Generating all available tiles n distance away from pos
        # t = itertools.product(range(pos[0]-n, pos[0]+n+1), range(pos[1]-n, pos[1]+n+1))
        # new_moves = filter(self.is_unoccupied, t)
        new_moves: list = list()
        # for i in t:
        #     if self.is_unoccupied(i):
        #         new_moves.append(i)
        i: cython.int
        j: cython.int
        tmp: (cython.int, cython.int)
        for i in range(pos[0]-n, pos[0]+n+1):
            for j in range(pos[1]-n, pos[1]+n+1):
                tmp = (i, j)
                if self.is_unoccupied(tmp):
                    new_moves.append(tmp)
        self.AI_moves = self.AI_moves.union(new_moves)

    # check for win condition
    @cython.cfunc
    def _check_win(self) -> cython.void:
        if self.turn_count < self.COUNT*2 - 1:  # Winning is not possible at this point
            return
        pos: (cython.int, cython.int) = self.prev_move
        self.game_ended = self._check_diagonal(pos) or self._check_vertical(pos) or self._check_horizontal(pos)
        if self.game_ended:
            self.game_state = self.turn
            # print(CHAR[self.game_state], "WON in", self.turn_count, "turns")
        else:
            if self.turn_count == self.size:
                self.game_ended = True
                # print("TIE")

    @cython.cfunc
    def _count_line(self, pos: (cython.int, cython.int), inc: (cython.int, cython.int)) -> cython.p_int:
        res: cython.int[2] = [0, False]
        while True:
            new_pos: (cython.int, cython.int) = pos[0] + inc[0], pos[1] + inc[1]
            if self.in_bound(new_pos):
                if self.board[pos[0]][pos[1]] == self.board[new_pos[0]][new_pos[1]]:  # if the line is still going
                    res[0] += 1
                    pos = new_pos
                else:
                    if self.board[new_pos[0]][new_pos[1]] == 0:  # if the line stops
                        return res
                    else:                         # if the line is blocked
                        res[1] = True
                        return res
            else:
                return res

    # Return True if a valid winning line going from pos to dir1 and dir2
    @cython.cfunc
    def _check_line(self, pos: (cython.int, cython.int), dir1: (cython.int, cython.int), dir2: (cython.int, cython.int)) -> cython.bint:
        line1: cython.int[2] = self._count_line(pos, dir1)
        line2: cython.int[2] = self._count_line(pos, dir2)
        length: cython.int = 1 + line1[0] + line2[0]
        blocked: cython.bint = line1[1] and line2[1]  # True if the line is blocked both ways
        if length > self.COUNT or (length == self.COUNT and not blocked):
            return True
        return False

    @cython.cfunc
    def _check_vertical(self, pos: (cython.int, cython.int)) -> cython.bint:
        return self._check_line(pos, UP, DOWN)

    @cython.cfunc
    def _check_horizontal(self, pos: (cython.int, cython.int)) -> cython.bint:
        return self._check_line(pos, RIGHT, LEFT)

    @cython.cfunc
    def _check_diagonal(self, pos: (cython.int, cython.int)) -> cython.bint:
        return self._check_line(pos, DOWN_RIGHT, UP_LEFT) or self._check_line(pos, UP_RIGHT, DOWN_LEFT)

    # simulate randomly from the current game state for n_turns, if n_turns == -1 simulate till the end
    @cython.ccall
    def simulate(self, n_turns: cython.int = -1):
        if n_turns == -1:
            while not self.game_ended:
                self.play(random.choice(list(self.AI_moves)))
        else:
            i: cython.int
            for i in range(n_turns):
                self.play(random.choice(list(self.AI_moves)))

    def __copy__(self) -> Caro:
        c: Caro = Caro(self.dim)
        c.turn_count = self.turn_count
        c.size = self.size
        c.board = np.copy(self.board)
        c.game_ended = self.game_ended
        c.game_state = self.game_state
        c.turn = self.turn
        c.prev_move = self.prev_move
        c.AI_moves = self.AI_moves.copy()
        return c

    def copy(self) -> Caro:
        return self.__copy__()

    @cython.ccall
    def copy_from(self, other: Caro) -> cython.void:
        self.turn_count = other.turn_count

        self.dim = other.dim
        self.size = other.size

        self.board = [[other.board[i][j] for j in range(30)] for i in range(30)] # Copy C array

        self.game_state = other.game_state
        self.game_ended = other.game_ended
        self.turn = other.turn
        self.prev_move = other.prev_move

        self.AI_moves = other.AI_moves.copy()


