# import numpy as np
import random
import time
import timeit
import cProfile
import itertools


DOWN = (1, 0)
UP = (-1, 0)
RIGHT = (0, 1)
LEFT = (0, -1)
DOWN_RIGHT = (1, 1)
DOWN_LEFT = (1, -1)
UP_RIGHT = (-1, 1)
UP_LEFT = (-1, -1)


class Caro:
    def __init__(self, dim):
        self.COUNT = 5
        self.turn_count = 0

        self.dim = int(dim)
        self.size = self.dim * self.dim
        # self.board = np.zeros((self.dim, self.dim))
        self.board = [[0 for _ in range(self.dim)] for i in range(self.dim)]

        self.game_state = 0     # 0:Undecided, 1: X wins, -1: Y wins
        self.game_ended = False
        self.turn = 1   # 1:X  -1:O
        self.prev_move = None
        self.char = {1: "X", -1: "O", 0: "."}

        self.AI_moves = {(int(self.dim/2), int(self.dim/2))}

    def __str__(self):
        res = ""
        for row in self.board:
            row_str = ""
            for i in row:
                row_str += self.char[i] + " "
            res += row_str + '\n'
        return res

    def in_bound(self, pos):
        return self.dim > pos[0] >= 0 and self.dim > pos[1] >= 0

    def is_unoccupied(self, pos):   # not using in_bound for speed
        return self.dim > pos[0] >= 0 and self.dim > pos[1] >= 0 and self.board[pos[0]][pos[1]] == 0

    def get_moves(self):
        return self.AI_moves

    # Place a piece at pos on board
    # turn argument for debugging
    def play(self, pos, turn=None):
        if self.game_ended:
            print("GAME ALREADY ENDED")
            return
        if not self.in_bound(pos):
            print(pos, "OUT OF BOUND")
            return False
        if self.board[pos[0]][pos[1]] == 0:
            self.board[pos[0]][pos[1]] = self.turn if turn is None else turn
            self.prev_move = pos
            self.turn_count += 1
            self._check_win()
            self._generate_AI_moves()
            self.turn = -self.turn      # switch turn
            return True
        else:
            print(tuple(pos), "already occupied")
            return False

    # FOR AI USAGE, give AI a set of more limited valid moves
    # generate moves that are at most n distance away from prev_pos
    def _generate_AI_moves(self, n=2):
        pos = self.prev_move
        self.AI_moves.discard(pos)
        # Generating all available tiles n distance away from pos
        # t = itertools.product(range(pos[0]-n, pos[0]+n+1), range(pos[1]-n, pos[1]+n+1))
        # new_moves = filter(self.is_unoccupied, t)
        new_moves = list()
        # for i in t:
        #     if self.is_unoccupied(i):
        #         new_moves.append(i)
        for i in range(pos[0]-n, pos[0]+n+1):
            for j in range(pos[1]-n, pos[1]+n+1):
                if self.is_unoccupied((i, j)):
                    new_moves.append((i, j))
        self.AI_moves = self.AI_moves.union(new_moves)

    # check for win condition
    def _check_win(self):
        if self.turn_count < self.COUNT*2 - 1:  # Winning is not possible at this point
            return
        pos = self.prev_move
        self.game_ended = self._check_diagonal(pos) or self._check_vertical(pos) or self._check_horizontal(pos)
        if self.game_ended:
            self.game_state = self.turn
            # print(self.char[self.game_state], "WON in", self.turn_count, "turns")
        else:
            if self.turn_count == self.size:
                self.game_ended = True
                # print("TIE")

    # count how many pieces in a line starting from pos going inc direction (not counting pos)
    # return (line_length, if its blocked)
    # def _count_line(self, pos, inc):
    #     new_pos = pos[0] + inc[0], pos[1] + inc[1]
    #     if self.in_bound(new_pos):
    #         if self.board[pos[0]][pos[1]] == self.board[new_pos[0]][new_pos[1]]:  # if the line is still going
    #             res = self._count_line(new_pos, inc)
    #             return 1 + res[0], res[1]
    #         else:
    #             if self.board[new_pos[0]][new_pos[1]] == 0:  # if the line stops
    #                 return 0, False
    #             else:                         # if the line is blocked
    #                 return 0, True
    #     else:
    #         return 0, False
    def _count_line(self, pos, inc):
        res = [0, False]
        while True:
            new_pos = pos[0] + inc[0], pos[1] + inc[1]
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
    def _check_line(self, pos, dir1, dir2):
        line1 = self._count_line(pos, dir1)
        line2 = self._count_line(pos, dir2)
        length = 1 + line1[0] + line2[0]
        blocked = line1[1] and line2[1]  # True if the line is blocked both ways
        if length > self.COUNT or (length == self.COUNT and not blocked):
            return True
        return False

    def _check_vertical(self, pos):
        return self._check_line(pos, UP, DOWN)

    def _check_horizontal(self, pos):
        return self._check_line(pos, RIGHT, LEFT)

    def _check_diagonal(self, pos):
        return self._check_line(pos, DOWN_RIGHT, UP_LEFT) or self._check_line(pos, UP_RIGHT, DOWN_LEFT)

    def __copy__(self):
        c = Caro(self.dim)
        c.turn_count = self.turn_count
        c.board = self.board.copy()
        c.game_ended = self.game_ended
        c.game_state = self.game_state
        c.turn = self.turn
        c.prev_move = self.prev_move
        c.AI_moves = self.AI_moves.copy()
        return c

    def copy(self):
        return self.__copy__()


def test(log=True):
    count = 1000
    win_count = 0
    tie_count = 0
    dim = 19
    t = time.time()
    # c = b.copy()
    for i in range(count):
        b = Caro(dim)
        while not b.game_ended:
            b.play(random.choice(list(b.get_moves())))
        if b.game_state == 1:
            win_count += 1
        elif b.game_state == 0:
            tie_count += 1
    if log:
        print(time.time() - t)
        print("X WON", win_count)
        print("O WON", count - win_count - tie_count)


def test2():
    dim = 30
    b = Caro(dim)
    while not b.game_ended:
        b.play(random.choice(list(b.get_moves())))
    print(b)


def test3(n):
    for i in range(n):
        test(False)


if __name__ == "__main__":
    n = 100
    # cProfile.run("test3(20)")
    # cProfile.run("test()")
    t = time.time()
    test3(n)
    print((time.time()-t)/n)
