from Caro_pybind import Caro, Point
from MCTS_pybind import MCTS_AI
from data_handler import save_raw_board, create_data_point, board_to_np, np_board_to_tensor, process_board, save_np_board
from model import initialize_SmallNetPybind
import time
import numpy as np
from copy import copy, deepcopy
# from pathos.helpers import mp
from torch import multiprocessing as mp
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


class GameMaster:
    def __init__(self, dim, count, AI1_params, AI2_params=None, self_play_mode=False,
                 verbose=False, eval_model=None, loss=mse_loss, reward_outcome=False,
                 num_workers=1):
        self.dim = dim
        self.count = count
        self.AI1_params = AI1_params
        self.AI2_params = AI2_params
        self.self_play_mode = self_play_mode
        if self.self_play_mode and self.AI2_params is not None:
            print("self_play_mode is True but AI2 params is not None, proceed as if self_play_mode=False")
            self.self_play_mode = False

        self.verbose = verbose
        self.eval_model = eval_model
        self.loss = loss

        self.reward_outcome = reward_outcome  # if true label target is the outcome of the game instead of mcts eval

        self.result = {1: 0, -1: 0, 0: 0}   # 1:X -1:O 0:TIE
        self.total_game_count = 0
        self.data_points = list()

        self.num_workers = num_workers
        self.processes = list()
        self.queue = None

    @staticmethod
    def init_self_play_worker(play_count, dim, count, AI1_params, AI2_params, self_play_mode,
                              reward_outcome, queue, worker_id):
        self_play_worker = SelfPlay(dim, count, AI1_params, AI2_params, self_play_mode,
                                    reward_outcome=reward_outcome,
                                    queue=queue, worker_id=worker_id)
        self_play_worker.start(play_count)

    def update_result(self, result):
        self.result[1] += result[1]  # win
        self.result[-1] += result[-1]  # loss
        self.result[0] += result[0]  # tie

    @staticmethod
    def write_data_to_file(f, data_points):
        if f is None:
            return
        for d in data_points:
            save_np_board(f, d[1], d[0])

    def start(self, play_count, outfile_path=None):
        data_count = 0
        outfile_path = outfile_path
        outfile = None
        if outfile_path:
            outfile = open(outfile_path, 'w')

        # Multiprocessing
        if self.num_workers > 1:    # Start multiprocess for self-play
            self.queue = mp.SimpleQueue()
            distributed_play_count = int(play_count/self.num_workers)
            r = play_count - distributed_play_count * self.num_workers
            remaining_play_count = distributed_play_count + r
            for worker_id in range(1, self.num_workers + 1):
                worker_play_count = distributed_play_count if worker_id < self.num_workers else remaining_play_count
                process = mp.Process(target=GameMaster.init_self_play_worker, args=(worker_play_count,
                                                                                    self.dim, self.count,
                                                                                    self.AI1_params, self.AI2_params,
                                                                                    self.self_play_mode,
                                                                                    self.reward_outcome,
                                                                                    self.queue, worker_id))
                self.processes.append(process)
                process.start()
            retrieved_count = 0
            terminated_workers = set()
            while retrieved_count < play_count:
                for i in range(len(self.processes)):
                    if (i not in terminated_workers) and (self.processes[i].exitcode is not None):
                        if self.processes[i].exitcode != 0:
                            print("WORKER_ID", i, "TERMINATED WITH CODE", self.processes[i].exitcode)
                        terminated_workers.add(i)
                if len(terminated_workers) >= self.num_workers:
                    break
                while self.queue is not None and not self.queue.empty():
                    # print("Fetching...", flush=True)
                    msg = self.queue.get()
                    if msg is None:
                        # worker_count += 1
                        # print(worker_count, "Workers done!")
                        continue
                    data = msg["data"]
                    self.data_points.extend(data)
                    data_count += len(data)
                    GameMaster.write_data_to_file(outfile, data)
                    self.result[msg["result"]] += 1
                    retrieved_count += 1
                    # print("Retrieved", retrieved_count, "games", flush=True)

        # No Multiprocessing
        elif self.num_workers <= 1:
            self_play = SelfPlay(self.dim, self.count, self.AI1_params, self.AI2_params, self.self_play_mode,
                                 verbose=self.verbose, reward_outcome=self.reward_outcome)
            self_play.start(play_count)
            self.data_points.extend(self_play.data_points)
            GameMaster.write_data_to_file(outfile, self_play.data_points)
            data_count += len(self_play.data_points)
            self.update_result(self_play.result)

        if outfile:
            outfile.write(str(len(self.data_points)) + "\n")
            outfile.close()

        for p in self.processes:
            p.join()
        for p in self.processes:
            if p.is_alive():
                p.terminate()

        print(self.result)
        # if self.loss and self.eval_model:
        #     avg_loss = self.loss(self.preds, self.labels)
        #     print("AVERAGE LOSS OVER", len(self.preds), "MOVE:", avg_loss)


class SelfPlay:
    def __init__(self, dim, count, AI1_params, AI2_params=None, self_play_mode=False,
                 outfile_path=None, verbose=False, eval_model=None, loss=mse_loss, reward_outcome=False,
                 queue=None, worker_id=0):
        self.dim = dim
        self.count = count
        self.AI1_params = AI1_params
        self.AI2_params = AI2_params
        self.self_play_mode = self_play_mode
        if self.self_play_mode and self.AI2_params is not None:
            print("self_play_mode is True but AI2 params is not None, proceed as if self_play_mode=False")
            self.self_play_mode = False
        if not self.self_play_mode and self.AI2_params is None:
            self.AI2_params = self.AI1_params
        self.caro_board = None

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

    def play(self, AI, audience_AI=None):
        t = time.time()
        move = AI.get_move(self.caro_board.get_prev_move())
        self.caro_board.play(move)
        AI.play_move(move)
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

        # print("PLAY worker", self.worker_id, time.time() - t, "SECONDS", flush=True)

        if not self.reward_outcome:
            self.data_points.append(create_data_point(AI.predicted_reward(), self.caro_board.get_board(), self.dim))
            if self.outfile:
                save_raw_board(self.outfile, AI.predicted_reward(), self.caro_board.get_board(), self.dim)
        else:
            self.board_history.append(self.caro_board.get_board())
        self.data_count += 1
        # switch player for AI in self-play mode
        if self.self_play_mode:
            AI.switch_player()

    # Store all [board, outcome] pair from a game, reset self.board_history
    def store_board_outcome(self, outcome):
        for b in self.board_history:
            self.data_points.append(create_data_point(outcome, b, self.dim))
            if self.outfile:
                save_raw_board(self.outfile, outcome, b, self.dim)
        self.board_history = list()

    # push self.data_points and result to queue, then clean self.data_points
    def push_data_to_queue(self, game_result):
        self.queue.put({"data": self.data_points, "result": game_result})
        self.data_points = list()

    def initialize_AI(self, AI_params, player, noise=0):
        random_threshold = AI_params["random_threshold"] + noise
        mcts_ai = MCTS_AI(player, AI_params["min_visit"], AI_params["n_sim"], self.caro_board,
                          _ai_moves_range=AI_params["AI_move_range"],
                          _eval=AI_params["eval"], _mode=AI_params["mode"],
                          _random_threshold=random_threshold)
        if "model" in AI_params:
            mcts_ai.initialize_model(AI_params["model"])
        if "uct_temperature" in AI_params:
            mcts_ai.set_uct_temperature(AI_params["uct_temperature"])
        if "random_transform" in AI_params and AI_params["random_transform"] is True:
            mcts_ai.enable_random_transform()
        else:
            mcts_ai.disable_random_transform()
        if "rollout_weight" in AI_params:
            mcts_ai.set_rollout_weight(AI_params["rollout_weight"])
        return mcts_ai

    def _self_play(self, i=0):
        total_t = time.time()

        temp_noise = -random.randint(0, 1)
        mcts_ai = self.initialize_AI(self.AI1_params, player=1, noise=temp_noise)
        if not self.self_play_mode:
            mcts_ai2 = self.initialize_AI(self.AI2_params, player=-1, noise=temp_noise)
        else:
            mcts_ai2 = mcts_ai

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

        if self.queue is not None:
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
        for AI_params in (self.AI1_params, self.AI2_params):
            if AI_params is None:
                continue
            if AI_params["model_param"] is not None:
                param = AI_params["model_param"]
                AI_params["model"] = initialize_SmallNetPybind(param)
                AI_params["eval"] = None
        # print("Start playing", play_count, "games| worker_id", self.worker_id, flush=True)
        game_count = 0
        for i in range(play_count):
            self.caro_board = Caro(self.dim, self.count)
            self.caro_board.disable_print()
            self._self_play(i)
            game_count += 1
            # print("GAME DONE " + str(i) + " worker_id " + str(self.worker_id), flush=True)
        # print("GAME DONE " + str(game_count) + " |worker_id " + str(self.worker_id), flush=True)

        if self.outfile:
            self.outfile.write(str(self.data_count) + "\n")
            self.outfile.close()

        # Finish signal
        if self.queue is not None:
            self.queue.put(None)


def play_eval(dim, count, play_count, param1, param2, swap=True, num_workers=0):
    AI1_params = param1
    AI2_params = param2
    AI1_name = "AI1 " if "name" not in param1 else param1["name"]
    AI2_name = "AI2 " if "name" not in param2 else param2["name"]
    result = list()
    for _ in range(1 + int(swap)):
        game_master = GameMaster(dim, count, AI1_params, AI2_params, num_workers=num_workers,
                                 verbose=False, reward_outcome=True)
        t = time.time()
        game_master.start(play_count)
        result.append(game_master.result)
        print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
        # swap AI
        AI1_params, AI2_params = AI2_params, AI1_params

    AI1_win = (result[0][1] + result[1][-1]) if swap else result[0][1]
    AI2_win = (result[0][-1] + result[1][1]) if swap else result[0][-1]
    tie_count = result[0][0] + result[1][0] if swap else result[0][0]
    print(f"{AI1_name} WON:", AI1_win)
    print(f"{AI2_name} WON:", AI2_win)
    print("TIE:", tie_count)
    return AI1_win, AI2_win, tie_count


# return [score, n_sim] AI1 evaluation against AI2
# n_sim is the highest n_sim AI2 with score >= SCORE_THRESHOLD
# Binary search for the highest n_sim
def baseline_eval(dim, count, play_count, AI1_params, AI2_params,
                  init_n_sim, min_sim, max_sim=-1, SCORE_THRESHOLD=0.45,
                  swap=True, AI1_as_X=True,
                  num_workers=0):
    n_sim = init_n_sim
    current_best_score = [0, 0]
    # Binary Search the most competitive n_sim
    while current_best_score[0] != n_sim * 100:
        AI2_params["n_sim"] = n_sim * 100
        print(f"Test {AI1_params['n_sim']} sims {AI1_params['name']} against {AI2_params['n_sim']} sims {AI2_params['name']}")
        if AI1_as_X:
            print("AI1 as X")
            result = play_eval(dim, count, play_count, AI1_params, AI2_params, swap=swap, num_workers=num_workers)
        else:
            print("AI1 as O")
            result = play_eval(dim, count, play_count, AI2_params, AI1_params, swap=swap, num_workers=num_workers)
        win_count = result[0] if AI1_as_X else result[1]
        score = (win_count + result[2] * 0.5) / sum(result)
        print("SCORE:", score)
        print()
        if score >= SCORE_THRESHOLD:
            current_best_score[1] = score
            current_best_score[0] = n_sim * 100
            min_sim = n_sim  # new min_sim
            if n_sim >= max_sim:
                n_sim = n_sim * 2
                max_sim = n_sim  # new max_sim
            else:
                new_sim = int((n_sim + max_sim) / 2)
                if new_sim == n_sim:
                    break
                n_sim = new_sim
        else:
            max_sim = n_sim  # new max_sim
            new_sim = int((min_sim + n_sim) / 2)
            if new_sim == n_sim:
                break
            n_sim = new_sim
    if current_best_score[0] == 0:
        current_best_score[0] = n_sim * 100
        current_best_score[1] = score
    return current_best_score


class SelfPlayGeneratePairBoardStates:
    def __init__(self, dim, count, n_sim, min_visits, AI_move_range,
                 outfile_path=None, verbose=False, queue=None, worker_id=0):
        self.dim = dim
        self.count = count
        self.n_sim = n_sim
        self.min_visits = min_visits
        self.AI_move_range = AI_move_range
        self.caro_board = None

        self.data_points = list()

        self.verbose = verbose

        self.worker_id = worker_id
        self.queue = queue
        self.processes = list()

        self.outfile_path = outfile_path
        self.outfile = None
        if self.outfile_path:
            self.outfile = open(self.outfile_path, 'a')

    def process_move(self, AI):
        return AI.get_move(self.caro_board.get_prev_move())

    # push self.data_points, then clean self.data_points
    def push_data_to_queue(self):
        self.queue.put(self.data_points)
        self.data_points = list()

    def _self_play(self, turn_count=0):
        total_t = time.time()

        random_threshold = 6
        mcts_ai = MCTS_AI(1, self.min_visits, self.n_sim, self.caro_board,
                          _ai_moves_range=self.AI_move_range,
                          _mode="alpha_zero",
                          _random_threshold=random_threshold)

        mcts_ai2 = MCTS_AI(-1, self.min_visits, self.n_sim, self.caro_board,
                           _ai_moves_range=self.AI_move_range,
                           _mode="alpha_zero",
                           _random_threshold=random_threshold)

        current_AI = mcts_ai if self.caro_board.current_player() == 1 else mcts_ai2
        current_player = self.caro_board.current_player()

        while not self.caro_board.has_ended():
            move = self.process_move(current_AI)
            if self.caro_board.get_turn_count() >= turn_count:
                break
            self.caro_board.play(move)
            if self.caro_board.has_ended():
                self.caro_board.undo()
                break
            current_AI.play_move(move)
            current_AI = mcts_ai if current_AI is mcts_ai2 else mcts_ai2
            current_player = -current_player

        max_i = current_AI.get_current_node_children_count()
        children_indexes = np.random.choice(max_i, size=2, replace=False)
        boards = []
        rewards = []
        pair_board_reward = []
        for i in children_indexes:
            m = current_AI.get_current_node_child_move(i)
            b = copy(self.caro_board)
            b.play(m)
            boards.append(b)
            r = current_AI.get_current_node_child_average_reward(i)
            rewards.append(r)
            pair_board_reward.append(create_data_point(r, b.get_board(), self.dim))

        self.data_points.extend(pair_board_reward)

        if self.outfile:
            for d in pair_board_reward:
                save_np_board(self.outfile, d[1], d[0])

        if self.queue is not None:
            self.push_data_to_queue()

        if self.verbose:
            print(current_player)
            for i in range(len(boards)):
                print(boards[i])
                print(rewards[i])
                print(pair_board_reward[i])
            print("-----")
            print("STATE GENERATED IN", time.time() - total_t, "SECONDS")
            print("AI1 has average", mcts_ai.average_child_count(), "children per expanded node")
            print("AI1 has average", mcts_ai2.average_child_count(), "children per expanded node")
            print("-----------------------------------------")

    def start(self, play_count):
        for i in range(play_count):
            turn_count = random.randint(self.count + 1, (self.dim - 1)**2 - self.count)
            self.caro_board = Caro(self.dim, self.count)
            self.caro_board.disable_print()
            self._self_play(turn_count)

        if self.outfile:
            self.outfile.write(str(play_count * 2) + "\n")
            self.outfile.close()


class GameMasterGeneratePairBoardStates:
    def __init__(self, dim, count, n_sim, min_visits, AI_move_range,
                 verbose=False, num_workers=1):
        self.dim = dim
        self.count = count
        self.n_sim = n_sim
        self.min_visits = min_visits
        self.AI_move_range = AI_move_range

        self.verbose = verbose

        self.total_game_count = 0
        self.data_points = list()

        self.num_workers = num_workers
        self.processes = list()
        self.queue = None

    @staticmethod
    def init_self_play_worker(play_count, dim, count,
                              n_sim, min_visits, AI_move_range,
                              queue, worker_id):
        self_play_worker = SelfPlayGeneratePairBoardStates(dim, count, n_sim, min_visits, AI_move_range,
                                                           queue=queue, worker_id=worker_id)
        self_play_worker.start(play_count)

    @staticmethod
    def write_data_to_file(f, data_points):
        if f is None:
            return
        for d in data_points:
            save_np_board(f, d[1], d[0])

    def start(self, play_count, outfile_path=None):
        data_count = 0
        outfile_path = outfile_path
        outfile = None
        if outfile_path:
            outfile = open(outfile_path, 'w')

        # Multiprocessing
        if self.num_workers > 1:    # Start multiprocess for self-play
            self.queue = mp.Queue()
            distributed_play_count = int(play_count/self.num_workers)
            r = play_count - distributed_play_count * self.num_workers
            remaining_play_count = distributed_play_count + r
            for worker_id in range(1, self.num_workers + 1):
                worker_play_count = distributed_play_count if worker_id < self.num_workers else remaining_play_count
                process = mp.Process(target=GameMasterGeneratePairBoardStates.init_self_play_worker,
                                     args=(worker_play_count, self.dim, self.count,
                                           self.n_sim, self.min_visits, self.AI_move_range,
                                           self.queue, worker_id))
                self.processes.append(process)
                process.start()

            retrieved_count = 0
            while retrieved_count < play_count:
                while self.queue is not None and not self.queue.empty():
                    data = self.queue.get()
                    self.data_points.extend(data)
                    data_count += len(data)
                    GameMasterGeneratePairBoardStates.write_data_to_file(outfile, data)
                    retrieved_count += 1

        # No Multiprocessing
        elif self.num_workers <= 1:
            self_play = SelfPlayGeneratePairBoardStates(self.dim, self.count,
                                                        self.n_sim, self.min_visits, self.AI_move_range,
                                                        verbose=self.verbose)
            self_play.start(play_count)
            self.data_points.extend(self_play.data_points)
            GameMasterGeneratePairBoardStates.write_data_to_file(outfile, self_play.data_points)
            data_count += len(self_play.data_points)

        if outfile:
            outfile.write(str(len(self.data_points)) + "\n")
            outfile.close()

        for p in self.processes:
            p.join()
        for p in self.processes:
            if p.is_alive():
                p.terminate()
