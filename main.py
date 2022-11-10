from Caro_pybind import Caro, Point
from MCTS_pybind import MCTS_AI, TreeNode
import time
import math
import torch

CHAR = {-1: "O", 0: ".", 1: "X"}


def str_board(_board):
    res = ""
    row_str = ""
    for i in range(dim):
        row_str = ""
        for j in range(dim):
            row_str += CHAR[_board[i][j]] + " "
        res += row_str + '\n'
    return res


def evaluate(_board):
    # print(str_board(_board))
    return 0.5


def save_data_point(f, _mcts_ai, _board):
    winrate = _mcts_ai.predicted_winrate()
    player = _mcts_ai.get_player()
    if player < 0:
        player = 2
    res = str(player) + "\n"
    board_array = _board.get_board()
    for i in range(board.get_dim()):
        for j in range(board.get_dim()):
            tmp = board_array[i][j]
            if tmp < 0:
                tmp = 2
            res += str(tmp) + " "
        res += "\n"
    res += str(winrate) + "\n\n"
    f.write(res)


if __name__ == "__main__":
    dir_path = "training_data/pass1.txt"
    outfile = open(dir_path, "a")

    dim = 15
    n_loops = 20
    n_sim1 = 20000
    min_visit1 = 20
    n_sim2 = 20000
    min_visit2 = 10

    x_win = 0
    data_count = 0

    for i in range(n_loops):
        board = Caro(dim)
        board.disable_print()
        mcts_ai = MCTS_AI(1, min_visit1, n_sim1, board, _eval=None)
        mcts_ai2 = MCTS_AI(-1, min_visit2, n_sim2, board, _eval=None)
        while not board.has_ended():
            t = time.time()
            board.play(mcts_ai.get_move(board.get_prev_move()))
            print(time.time()-t, "SECONDS")
            print("DEPTH:", mcts_ai.get_tree_depth())
            print("X PLAYED", board.get_prev_move(), "with predicted winrate", mcts_ai.predicted_winrate())
            print(board)
            save_data_point(outfile, mcts_ai, board)
            data_count += 1

            if board.has_ended():
                continue

            t = time.time()
            board.play(mcts_ai2.get_move(board.get_prev_move()))
            print(time.time() - t, "SECONDS")
            print("DEPTH:", mcts_ai2.get_tree_depth())
            print("O PLAYED", board.get_prev_move(), "with predicted winrate", mcts_ai2.predicted_winrate())
            print(board)
            save_data_point(outfile, mcts_ai2, board)
            data_count += 1

        if board.get_state() == 1:
            print("X WON")
            x_win += 1
        elif board.get_state() == -1:
            print("O WON")
        else:
            print("TIE")

        # print("AI1 has average", mcts_ai.average_child_count(), "children per expanded node")
        # print("AI1 has average", mcts_ai2.average_child_count(), "children per expanded node")
        # print(board)
        print("------------------------------------------------------------------------\n")

    print(x_win/n_loops)
    print(data_count)
    outfile.write(str(data_count) + "\n")
    outfile.close()
