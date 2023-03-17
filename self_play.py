from Caro_pybind import Caro, Point
from MCTS_pybind import MCTS_AI
from MCTS import MCTS
from data_handler import save_raw_data, create_data_point, board_to_np, np_board_to_tensor, process_board
from model import optimize_model, SmallNet
import time
import numpy as np
import torch
# from pathos.helpers import mp
import torch.multiprocessing as mp
import random


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


class SelfPlay:
    def __init__(self, dim, count, AI1_params, AI2_params,
                 outfile_path=None, verbose=False, eval_model=None, loss=mse_loss, reward_outcome=False,
                 num_workers=1, queue=None, worker_id=0, torch_optimize=False):
        self.dim = dim
        self.count = count
        self.AI1_params = AI1_params
        self.AI2_params = AI2_params
        self.caro_board = None
        self.torch_optimize = torch_optimize

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

        self.worker_id = worker_id
        self.queue = queue
        self.num_workers = num_workers
        self.processes = list()

        self.outfile_path = outfile_path
        self.outfile = None
        if self.outfile_path:
            self.outfile = open(self.outfile_path + "-" + str(self.worker_id) + ".txt", 'a')

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

        print("PLAY worker", self.worker_id, time.time() - t, "SECONDS", flush=True)

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

    @staticmethod
    def init_self_play_worker(play_count, dim, count, AI1_params, AI2_params,
                              outfile_path, reward_outcome, queue, worker_id):
        self_play_worker = SelfPlay(dim, count, AI1_params, AI2_params,
                                    outfile_path, reward_outcome=reward_outcome,
                                    queue=queue, worker_id=worker_id)
        print("Initialized worker " + str(worker_id), flush=True)
        self_play_worker.start(play_count)

    # push self.data_points and result to queue, then clean self.data_points
    def push_data_to_queue(self, game_result):
        self.queue.put({"data": self.data_points, "result": game_result})
        self.data_points = list()

    def handle_queue(self):
        pass

    def _self_play(self, i=0):
        # print("GAME", i+1)
        total_t = time.time()
        self.caro_board = Caro(self.dim, self.count)
        self.caro_board.disable_print()
        random_threshold1 = self.AI1_params["random_threshold"] #+ random.randint(0, 1)
        random_threshold2 = self.AI2_params["random_threshold"] #+ random.randint(0, 1)
        mcts_ai = MCTS_AI(1, self.AI1_params["min_visit"], self.AI1_params["n_sim"], self.caro_board,
                          _ai_moves_range=self.AI1_params["AI_move_range"],
                          _eval=self.AI1_params["eval"], _mode=self.AI1_params["mode"],
                          _random_threshold=random_threshold1,)
        mcts_ai2 = MCTS_AI(-1, self.AI2_params["min_visit"], self.AI2_params["n_sim"], self.caro_board,
                           _ai_moves_range=self.AI1_params["AI_move_range"],
                           _eval=self.AI2_params["eval"], _mode=self.AI2_params["mode"],
                           _random_threshold=random_threshold2,)

        while not self.caro_board.has_ended():
            self.play(mcts_ai)
            if self.caro_board.has_ended():
                break
            self.play(mcts_ai2)

        game_result = self.caro_board.get_state()
        if self.reward_outcome:
            self.store_board_outcome(game_result)
        self.total_game_count += 1
        self.result[game_result] += 1

        if self.worker_id > 0 and self.queue is not None:
            self.push_data_to_queue(game_result)

        if self.verbose:
            if game_result == 1:
                print("X WON")
            elif game_result == -1:
                print("O WON")
            else:
                print("TIE")
            print(self.caro_board)
            print("GAME ENDED IN", time.time() - total_t, "SECONDS")
            print("AI1 has average", mcts_ai.average_child_count(), "children per expanded node")
            print("AI1 has average", mcts_ai2.average_child_count(), "children per expanded node")
            print("-----------------------------------------")

    def start(self, play_count):
        total_play_count = play_count
        if self.num_workers > 1:    # Start multiprocess for self-play
            self.queue = mp.SimpleQueue()
            distributed_play_count = int(total_play_count/self.num_workers)
            r = total_play_count - distributed_play_count * self.num_workers
            play_count = distributed_play_count + r
            for worker_id in range(1, self.num_workers):
                process = mp.Process(target=self.init_self_play_worker, args=(distributed_play_count,
                                                                              self.dim, self.count,
                                                                              self.AI1_params, self.AI2_params,
                                                                              self.outfile_path, self.reward_outcome,
                                                                              self.queue, worker_id))
                self.processes.append(process)
                process.start()
            print("Multiprocess started.")

        # print("INITIALIZING AI worker", self.worker_id)
        # Initialize AI
        for AI_params in (self.AI1_params, self.AI2_params):
            if AI_params["model_state_dict"] is not None:
                model = SmallNet()
                print("LOADING MODEL FROM STATE_DICT, worker", self.worker_id)
                model.load_state_dict(AI_params["model_state_dict"])
                print("MODEL LOADED, worker", self.worker_id)
                if self.torch_optimize:
                    model = optimize_model(model)
                AI_params["eval"] = get_evaluate_function(model)
        # print("INITIALIZED AI worker", self.worker_id)

        retrieved_count = 0
        for i in range(play_count):
            self._self_play(i)
            print("GAME DONE " + str(i) + " worker_id " + str(self.worker_id), flush=True)
            if self.worker_id == 0 and self.num_workers > 1:
                while self.queue is not None and not self.queue.empty():
                    msg = self.queue.get()
                    self.data_points.extend(msg["data"])
                    self.result[msg["result"]] += 1
                    retrieved_count += 1
                    # print("GAME " + str(i + retrieved_count) + " retrieved")

        if self.worker_id > 0:
            return

        while retrieved_count + play_count < total_play_count:
            while self.queue is not None and not self.queue.empty():
                msg = self.queue.get()
                self.data_points.extend(msg["data"])
                self.result[msg["result"]] += 1
                retrieved_count += 1

        for p in self.processes:
            p.join()

        print(self.result)
        if self.loss and self.eval_model:
            avg_loss = self.loss(self.preds, self.labels)
            print("AVERAGE LOSS OVER", len(self.preds), "MOVE:", avg_loss)
        if self.outfile:
            self.outfile.write(str(self.data_count) + "\n")
        print("==========================================================================================\n")
        return self.data_points
