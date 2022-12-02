from Caro_pybind import Caro, Point
from MCTS_pybind import MCTS_AI
from data_handler import save_raw_data, create_data_point, board_to_np, np_board_to_tensor, process_board
import time


def get_evaluate_function(model):
    def evaluate(board, dim):
        board = process_board(board, dim)
        board = board_to_np(board, dim=len(board))
        # print(board)
        board = np_board_to_tensor(board, unsqueeze=True)
        res = model(board)
        # print(res)
        return res
    return evaluate


# Return data points from self-play
def self_play(dim, count, play_count, n_sim1, min_visit1, eval1, mode1, n_sim2, min_visit2, eval2, mode2, outfile=None,
              verbose=False, eval_model=None):
    x_win = 0
    data_count = 0
    data_points = list()
    for i in range(play_count):
        total_t = time.time()
        caro_board = Caro(dim, count)
        caro_board.disable_print()
        mcts_ai = MCTS_AI(1, min_visit1, n_sim1, caro_board, _eval=eval1, _mode=mode1)
        mcts_ai2 = MCTS_AI(-1, min_visit2, n_sim2, caro_board, _eval=eval2, _mode=mode2)
        while not caro_board.has_ended():
            t = time.time()
            caro_board.play(mcts_ai.get_move(caro_board.get_prev_move()))
            if verbose:
                print(time.time() - t, "SECONDS")
                print("DEPTH:", mcts_ai.get_tree_depth())
                print("X PLAYED", caro_board.get_prev_move(), "with predicted reward", mcts_ai.predicted_reward())
                if eval_model:
                    pred = float(eval_model(caro_board.get_board(), caro_board.get_dim()))
                    print("MODEL REWARD PREDICTION:", pred)
                print(caro_board)
            data_points.append(create_data_point(mcts_ai, caro_board))
            if outfile:
                save_raw_data(outfile, mcts_ai, caro_board)
            data_count += 1

            if caro_board.has_ended():
                continue

            t = time.time()
            caro_board.play(mcts_ai2.get_move(caro_board.get_prev_move()))
            if verbose:
                print(time.time() - t, "SECONDS")
                print("DEPTH:", mcts_ai2.get_tree_depth())
                print("O PLAYED", caro_board.get_prev_move(), "with predicted reward", mcts_ai2.predicted_reward())
                if eval_model:
                    pred = float(eval_model(caro_board.get_board(), caro_board.get_dim()))
                    print("MODEL REWARD PREDICTION:", pred)
                print(caro_board)
            data_points.append(create_data_point(mcts_ai2, caro_board))
            if outfile:
                save_raw_data(outfile, mcts_ai2, caro_board)
            data_count += 1

        if caro_board.get_state() == 1:
            print("X WON")
            x_win += 1
        elif caro_board.get_state() == -1:
            print("O WON")
        else:
            print("TIE")
        print(caro_board)
        print("GAME ENDED IN", time.time()-total_t, "SECONDS")
        if verbose:
            print("AI1 has average", mcts_ai.average_child_count(), "children per expanded node")
            print("AI1 has average", mcts_ai2.average_child_count(), "children per expanded node")
        print("------------------------------------------------------------------------\n")

    if verbose:
        print(x_win / play_count)
    if outfile:
        outfile.write(str(data_count) + "\n")
    return data_points

