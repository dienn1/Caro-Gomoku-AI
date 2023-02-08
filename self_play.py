from Caro_pybind import Caro, Point
from MCTS_pybind import MCTS_AI
from data_handler import save_raw_data, create_data_point, board_to_np, np_board_to_tensor, process_board
import time
import numpy as np


def get_evaluate_function(model):
    def evaluate(board, dim):
        board = process_board(board, dim)
        board = board_to_np(board, dim=len(board))
        board = np_board_to_tensor(board, unsqueeze=True)
        res = model(board)
        return res

    return evaluate


def mse_loss(pred, label, axis=None):
    pred = np.array(pred)
    label = np.array(label)
    loss = (np.square(pred - label)).mean(axis=axis)
    return loss


# Return data points from self-play (deprecated)
def self_play(dim, count, play_count, n_sim1, min_visit1, eval1, mode1, n_sim2, min_visit2, eval2, mode2, outfile=None,
              verbose=False, eval_model=None, loss=mse_loss):
    x_win = 0
    data_count = 0
    data_points = list()
    preds = list()
    labels = list()
    for i in range(play_count):
        total_t = time.time()
        caro_board = Caro(dim, count)
        caro_board.disable_print()
        mcts_ai = MCTS_AI(1, min_visit1, n_sim1, caro_board, _eval=eval1, _mode=mode1)
        mcts_ai2 = MCTS_AI(-1, min_visit2, n_sim2, caro_board, _eval=eval2, _mode=mode2)
        while not caro_board.has_ended():
            t = time.time()
            caro_board.play(mcts_ai.get_move(caro_board.get_prev_move()))
            label = mcts_ai.predicted_reward()
            pred = float(eval_model(caro_board.get_board(), caro_board.get_dim()))
            if loss:
                labels.append(label)
                preds.append(pred)
            if verbose:
                print(time.time() - t, "SECONDS")
                print("DEPTH:", mcts_ai.get_tree_depth())
                print("X PLAYED", caro_board.get_prev_move(), "with predicted reward", label)
                if eval_model:
                    print("MODEL REWARD PREDICTION:", pred)
                print(caro_board)
            data_points.append(create_data_point(mcts_ai.predicted_reward(), caro_board.get_board(), caro_board.get_dim()))
            if outfile:
                save_raw_data(outfile, mcts_ai.predicted_reward(), caro_board.get_board(), caro_board.get_dim())
            data_count += 1

            if caro_board.has_ended():
                continue

            t = time.time()
            caro_board.play(mcts_ai2.get_move(caro_board.get_prev_move()))
            label = mcts_ai2.predicted_reward()
            pred = float(eval_model(caro_board.get_board(), caro_board.get_dim()))
            if loss:
                labels.append(label)
                preds.append(pred)
            if verbose:
                print(time.time() - t, "SECONDS")
                print("DEPTH:", mcts_ai2.get_tree_depth())
                print("O PLAYED", caro_board.get_prev_move(), "with predicted reward", label)
                if eval_model:
                    print("MODEL REWARD PREDICTION:", pred)
                print(caro_board)
            data_points.append(create_data_point(mcts_ai2.predicted_reward(), caro_board.get_board(), caro_board.get_dim()))
            if outfile:
                save_raw_data(outfile, mcts_ai2.predicted_reward(), caro_board.get_board(), caro_board.get_dim())
            data_count += 1

        if caro_board.get_state() == 1:
            print("X WON")
            x_win += 1
        elif caro_board.get_state() == -1:
            print("O WON")
        else:
            print("TIE")
        print(caro_board)
        print("GAME ENDED IN", time.time() - total_t, "SECONDS")
        if verbose:
            print("AI1 has average", mcts_ai.average_child_count(), "children per expanded node")
            print("AI1 has average", mcts_ai2.average_child_count(), "children per expanded node")
        print("------------------------------------------------------------------------\n")

    if verbose:
        print(x_win / play_count)
    if loss:
        avg_loss = loss(preds, labels)
        print("AVERAGE LOSS OVER", len(preds), "MOVE:", avg_loss)
    if outfile:
        outfile.write(str(data_count) + "\n")
    return data_points


class SelfPlay:
    def __init__(self, dim, count, AI1_params, AI2_params,
                 outfile=None, verbose=False, eval_model=None, loss=mse_loss, reward_outcome=False):
        self.dim = dim
        self.count = count
        self.AI1_params = AI1_params
        self.AI2_params = AI2_params
        self.caro_board = None

        self.outfile = outfile
        self.verbose = verbose
        self.eval_model = eval_model
        self.loss = loss

        self.reward_outcome = reward_outcome  # if true label target is the outcome of the game instead of mcts eval
        self.board_history = list()

        self.result = {1: 0, -1: 0, 0: 0}   # 1:X -1:O 0:TIE
        self.data_count = 0
        self.total_game_count = 0
        self.data_points = list()
        self.preds = list()
        self.labels = list()

    @staticmethod
    def player_symbol(player):
        symbol = "."
        if player == 1:
            symbol = "X"
        elif player == -1:
            symbol = "O"
        return symbol

    def play(self, AI):
        t = time.time()
        self.caro_board.play(AI.get_move(self.caro_board.get_prev_move()))
        label = AI.predicted_reward()
        if self.eval_model:
            pred = float(self.eval_model(self.caro_board.get_board(), self.dim))
            if self.loss:
                self.labels.append(label)
                self.preds.append(pred)
        if self.verbose:
            print(time.time() - t, "SECONDS")
            print("DEPTH:", AI.get_tree_depth())
            print(self.player_symbol(AI.get_player()), "PLAYED", self.caro_board.get_prev_move(), "with predicted reward", label)
            if self.eval_model:
                print("MODEL REWARD PREDICTION:", pred)
            print(self.caro_board)
        if not self.reward_outcome:
            self.data_points.append(create_data_point(AI.predicted_reward(), self.caro_board.get_board(), self.dim))
            if self.outfile:
                save_raw_data(self.outfile, AI.predicted_reward(), self.caro_board.get_board(), self.dim)
        else:
            self.board_history.append(self.caro_board.get_board())
        self.data_count += 1

    # Store all [board, outcome] pair from a game, reset self.board_history
    def store_board_outcome(self, outcome):
        for b in self.board_history:
            self.data_points.append(create_data_point(outcome, b, self.dim))
            if self.outfile:
                save_raw_data(self.outfile, outcome, b, self.dim)
        self.board_history = list()

    def self_play(self, play_count):
        for i in range(play_count):
            print("GAME", i+1)
            total_t = time.time()
            self.caro_board = Caro(self.dim, self.count)
            self.caro_board.disable_print()
            mcts_ai = MCTS_AI(1, self.AI1_params["min_visit"], self.AI1_params["n_sim"], self.caro_board,
                              _ai_moves_range=self.AI1_params["AI_move_range"],
                              _eval=self.AI1_params["eval"], _mode=self.AI1_params["mode"])
            mcts_ai2 = MCTS_AI(-1, self.AI2_params["min_visit"], self.AI2_params["n_sim"], self.caro_board,
                               _ai_moves_range=self.AI1_params["AI_move_range"],
                               _eval=self.AI2_params["eval"], _mode=self.AI2_params["mode"])

            while not self.caro_board.has_ended():
                self.play(mcts_ai)
                if self.caro_board.has_ended():
                    continue
                self.play(mcts_ai2)

            game_result = self.caro_board.get_state()
            if self.reward_outcome:
                self.store_board_outcome(game_result)

            self.total_game_count += 1
            self.result[game_result] += 1
            if game_result == 1:
                print("X WON")
            elif game_result == -1:
                print("O WON")
            else:
                print("TIE")
            print(self.caro_board)
            print("GAME ENDED IN", time.time() - total_t, "SECONDS")
            if self.verbose:
                print("AI1 has average", mcts_ai.average_child_count(), "children per expanded node")
                print("AI1 has average", mcts_ai2.average_child_count(), "children per expanded node")
            print("-----------------------------------------")

        if self.verbose:
            print(self.result)
        if self.loss and self.eval_model:
            avg_loss = self.loss(self.preds, self.labels)
            print("AVERAGE LOSS OVER", len(self.preds), "MOVE:", avg_loss)
        if self.outfile:
            self.outfile.write(str(self.data_count) + "\n")
        print("==========================================================================================\n")
        return self.data_points
