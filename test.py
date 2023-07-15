import torch
from data_handler import load_raw_board_data, load_data, batch_load_data, BoardDataLoader
from model import SmallNet, train, load_model_from_file, FFNet, optimize_model, get_param
from self_play import GameMaster, GameMasterGeneratePairBoardStates, play_eval, baseline_eval
from data_handler import load_model_and_train, load_data_and_train, np_to_board, np_board_to_tensor_batch
import random
import time
import numpy as np
from Model_pybind import SmallNet as SmallNetPybind
from copy import deepcopy, copy
from torch import multiprocessing as mp
import json
import os

if os.name == "nt":
    import matplotlib.pyplot as plt


def str_board(board):
    res = ""
    dim = len(board)
    for i in range(dim):
        row_str = ""
        for j in range(dim):
            row_str += str(board[i][j]) + " "
        res += row_str + '\n'
    return res


if __name__ == "__main__":
    # t = 3600
    # print("SLEEPING FOR", t, "SECONDS")
    # time.sleep(t)

    mp.set_start_method("spawn")
    num_workers = 60

    dim = 7
    count = 5
    play_count = num_workers * 2
    # swap_size = False

    PATH = "training_data/"
    SUB_PATH = "Test7x7/"

    NN_NAME = "attempt22/ValueNet_model_400_360_pass15.pt"
    NN_PATH = PATH + SUB_PATH + NN_NAME
    nn_model = load_model_from_file(NN_PATH, SmallNet)

    NN_NAME2 = "attempt16/ValueNet_model_400_360_pass20.pt"
    NN_PATH2 = PATH + SUB_PATH + NN_NAME2
    nn_model2 = load_model_from_file(NN_PATH2, SmallNet)

    # data_path = PATH + SUB_PATH + "generated_pair_states_20000.txt"
    # data_path = PATH + SUB_PATH + "warm_up_1000.txt"
    # outfile_path = PATH + SUB_PATH + "self_play_eval_test.json"
    outfile_path = None
    # model = nn_model
    # model = SmallNet()
    # t = time.time()
    # data = load_data_and_train(data_path, model, data_count=5000, training_batch_size=2048,
    #                            lr=0.005, batch_size=32, total_epoch=100, weight_decay=0.001, transform_variance_count=2)
    # print("Loading data and training takes", time.time()-t)
    # print(len(data))
    # exit()
    # del data

    param1 = get_param(nn_model)
    param2 = get_param(nn_model2)

    AI_move_range = 1
    random_threshold = 4

    # AI1_params = {"n_sim": 400,
    #               "min_visit": 1,
    #               "name": "NN_MCTS",
    #               "AI_move_range": AI_move_range,
    #               "mode": "greedy",
    #               "random_threshold": random_threshold,
    #               "uct_temperature": 1,
    #               "rollout_weight": 0,
    #               "random_transform": False,
    #               "model_param": param1,
    #               "eval": None}
    # AI2_params = {"n_sim": 1000,
    #               "min_visit": 1,
    #               "name": "1_minvisit_vanilla",
    #               "AI_move_range": AI_move_range,
    #               "mode": "greedy",
    #               "random_threshold": random_threshold,
    #               "model_param": None,
    #               "eval": None}
    # num_workers = 4
    # play_count = num_workers * 20
    # swap_size = False
    # # print(f"Test with {AI1_params['n_sim']} sims {AI1_params['name']} against {AI2_params['n_sim']} sims {AI2_params['name']}")
    # print(f"Test with {AI1_params['n_sim']} sims {AI1_params['name']} self-play")
    # game_master = GameMaster(dim, count, AI1_params, AI2_params=None, self_play_mode=True,
    #                          num_workers=num_workers,
    #                          verbose=False, eval_model=None, reward_outcome=True)
    # print("Start", play_count, "games of self-play")
    # t = time.time()
    # game_master.start(play_count, outfile_path=outfile_path)
    # data = game_master.data_points
    # print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
    # print(len(data))
    # # result = play_eval(dim, count, play_count, AI1_params, AI2_params,
    # #                    swap=swap_size,
    # #                    num_workers=num_workers)
    # print("-----------------------")
    # exit()

    # for i in range(10, 18):
    #     n_sim = i * 100
    #     AI2_params["n_sim"] = n_sim
    #     print(f"Test with {n_sim} sims for vanilla MCTS against {AI1_params['n_sim']} sims " + NN_NAME)
    #     game_master = GameMaster(dim, count, AI1_params, AI2_params, num_workers=num_workers,
    #                              verbose=False, eval_model=None, reward_outcome=True)
    #     print("Start", play_count, "games of self-play")
    #     t = time.time()
    #     game_master.start(play_count, outfile_path=outfile_path)
    #     data = game_master.data_points
    #     print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
    #     print(len(data))
    #     print("-----------------------")

    # CHECKPOINT EVAL
    # AI1_params = {"n_sim": 200,
    #               "min_visit": 1,
    #               "AI_move_range": AI_move_range,
    #               "mode": "alpha_zero",
    #               "random_threshold": random_threshold,
    #               "model_param": None,
    #               "eval": None}
    # AI2_params = {"n_sim": 200,
    #               "min_visit": 1,
    #               "AI_move_range": AI_move_range,
    #               "mode": "alpha_zero",
    #               "random_threshold": random_threshold,
    #               "model_param": None,
    #               "eval": None}
    # passes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 149]
    # self_play_eval_results = {"__comment__": "evaluate each checkpoint model against each other",
    #                           "__comment2__": "n_sims = 200; random_threshold = 4; random_threshold randomly +1"}
    # for p in passes:
    #     self_play_eval_results["pass_" + str(p)] = list()
    # json_path = PATH + SUB_PATH + "attempt18/self_play_checkpoint_eval_randomthreshold.json"
    # for i in range(len(passes)):
    #     ai1_pass = passes[i]
    #     ai1_str = "pass_" + str(ai1_pass)
    #     nn_name = "attempt18/ValueNet_model_400_360_pass" + str(ai1_pass) + ".pt"
    #     nn_path = PATH + SUB_PATH + nn_name
    #     nn_model = load_model_from_file(nn_path, SmallNet)
    #     AI1_params["model_param"] = get_param(nn_model)
    #     for ai2_pass in passes[i:]:
    #         ai2_str = "pass_" + str(ai2_pass)
    #         nn_name2 = "attempt18/ValueNet_model_400_360_pass" + str(ai2_pass) + ".pt"
    #         nn_path2 = PATH + SUB_PATH + nn_name2
    #         nn_model2 = load_model_from_file(nn_path2, SmallNet)
    #         AI2_params["model_param"] = get_param(nn_model2)
    #         print(f"Test {nn_name} against {nn_name2}\nn_sims=200, random_threshold={random_threshold}")
    #         result = self_play_eval(dim, count, play_count, AI1_params, AI2_params,
    #                                 swap=swap_size,
    #                                 num_workers=num_workers)
    #         self_play_eval_results[ai1_str].append(result)      # add result for AI1 vs AI2
    #         if ai1_str != ai2_str:
    #             self_play_eval_results[ai2_str].append([result[1], result[0], result[2]])   # add result for AI2 vs AI1
    # with open(json_path, "w") as outfile:
    #     json.dump(self_play_eval_results, outfile, indent=4)
    # exit()

    uct_temperature = 1
    rollout_weight = 0
    # Eval Vanilla MCTS Score
    AI1_params = {"n_sim": 200,
                  "min_visit": 1,
                  "AI_move_range": AI_move_range,
                  "mode": "greedy",
                  "random_threshold": random_threshold,
                  "model_param": None,
                  "uct_temperature": uct_temperature,
                  "rollout_weight": rollout_weight,
                  "random_transform": False,
                  "name": None,
                  "eval": None}
    AI2_params = {"n_sim": 200,
                  "min_visit": 1,
                  "AI_move_range": AI_move_range,
                  "mode": "greedy",
                  "random_threshold": random_threshold,
                  "model_param": None,
                  "name": "1_minvisit_vanilla",
                  "eval": None}
    json_path = PATH + SUB_PATH + "attempt25/baseline_eval_no_rollout2.json"
    baseline_score = {"__comment__": "[n_sim, score] for each pass, n_sim being the highest n_sim Vanilla MCTS the AI can score > 0.45",
                      "__comment2__": "NN MCTS n_sim=200, vanilla MCTS min_visit=1, play_count=240, swap size, rollout_weight=0, uct_temperature=1"}
    passes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]
    AI1_as_X = True
    swap_size = True
    SCORE_THRESHOLD = 0.45
    min_sim = 3
    max_sim = -1
    init_n_sim = min_sim
    for p in passes:
        nn_name = "attempt25/ValueNet_model_400_360_pass" + str(p) + ".pt"
        nn_path = PATH + SUB_PATH + nn_name
        nn_model = load_model_from_file(nn_path, SmallNet)
        AI1_params["model_param"] = get_param(nn_model)
        AI1_params["name"] = nn_name
        score = baseline_eval(dim, count, play_count, AI1_params, AI2_params,
                              init_n_sim, min_sim, max_sim, SCORE_THRESHOLD,
                              swap=swap_size, AI1_as_X=AI1_as_X,
                              num_workers=num_workers)
        baseline_score[p] = score
        print(f"Pass {p} score: {score}")
        print("---------------\n")
        init_n_sim = int((score[0])/100) + 1  # Initialize n_sim as the last best n_sim + 100
    with open(json_path, "w") as outfile:
        json.dump(baseline_score, outfile, indent=4)

    # print(f"Test {NN_NAME} against {NN_NAME2}\nn_sims={AI1_params['n_sim']}, random_threshold={random_threshold}")
    # result = self_play_eval(dim, count, play_count, param1, param2,
    #                         AI_move_range=AI_move_range, random_threshold=random_threshold, num_workers=num_workers)

    # json_path = PATH + SUB_PATH + "attempt18/self_play_eval.json"
    # with open(json_path) as f:
    #     self_play_eval_dict = json.load(f)
    # self_play_eval_score = {'prev_eval': list(), 'prev_10_eval': list()}
    # for t in ['prev_eval', 'prev_10_eval']:
    #     for i in range(len(self_play_eval_dict[t])):
    #         res = self_play_eval_dict[t][i]
    #         score = (res[0] + res[2] * 0.5) / sum(res)
    #         self_play_eval_score[t].append(score)
    #     self_play_eval_score[t] = np.array(self_play_eval_score[t])
    #
    # plt.plot(range(0, 160), [0.5] * 160)
    # plt.plot(range(1, len(self_play_eval_dict['prev_eval']) + 1), self_play_eval_score['prev_eval'], color='b')
    # plt.xlabel("Training Pass")
    # plt.ylabel("Score")
    # plt.show()
    #
    # plt.plot(range(0, 160), [0.5] * 160)
    # plt.plot(range(11, len(self_play_eval_dict['prev_10_eval']) + 11), self_play_eval_score['prev_10_eval'], color='r')
    # plt.xlabel("Training Pass")
    # plt.ylabel("Score")
    # plt.show()
    #
    # x = self_play_eval_score['prev_eval']
    # print(len(x[x < 0.45]))
    # x = self_play_eval_score['prev_10_eval']
    # print(len(x[x < 0.45]))

    # GENERATE PAIR BOARD STATES
    # n_sim = 20000
    # print("Start generate Pair board states with n_sim=" + str(n_sim))
    # t = time.time()
    # game_master = GameMasterGeneratePairBoardStates(dim, count, n_sim=n_sim, min_visits=10, AI_move_range=AI_move_range,
    #                                                 verbose=False, num_workers=num_workers)
    # game_master.start(play_count=play_count, outfile_path=outfile_path)
    # print("Time for " + str(play_count) + " states pair generated:", time.time() - t)

    # TEST PAIR BOARD STATES EVAL
    # paired_states = (load_data(data_dir=data_path, data_count=99999)).data
    # pair_binary = list()
    # paired_states_board = list()
    # pair_diff = list()
    # for i in range(int(len(paired_states)/2)):
    #     diff = np.abs(paired_states[2*i][1] - paired_states[2*i + 1][1])
    #     if diff < 0.1:
    #         continue
    #     pair_diff.append(diff)
    #     paired_states_board.append(paired_states[2*i][0].detach().numpy())
    #     paired_states_board.append(paired_states[2*i + 1][0].detach().numpy())
    #     if paired_states[2*i][1] < paired_states[2*i + 1][1]:
    #         pair_binary.append(1)
    #     else:
    #         pair_binary.append(0)
    # pair_binary = np.array(pair_binary)
    # pair_diff = np.array(pair_diff)
    # # print(pair_binary)
    # print(np.mean(pair_binary))
    # print(pair_diff)
    # print(np.mean(pair_diff))
    # paired_states_board = np.array(paired_states_board)
    # print(paired_states_board.shape)
    # paired_states_board = torch.from_numpy(paired_states_board).type(torch.FloatTensor)
    #
    # paired_states_pred = nn_model(paired_states_board)
    # pair_binary_pred = list()
    # for i in range(len(pair_binary)):
    #     if paired_states_pred[2*i] < paired_states_pred[2*i + 1]:
    #         pair_binary_pred.append(1)
    #     else:
    #         pair_binary_pred.append(0)
    # pair_binary_pred = np.array(pair_binary_pred)
    # abs_diff = np.mean(np.abs(pair_binary_pred - pair_binary))
    # print(abs_diff)

    # test_data = (load_data(data_dir=data_path, data_count=99999)).data
    # input_board = np.array(list(d[0].detach().numpy() for d in test_data))
    # input_board = torch.from_numpy(input_board).type(torch.FloatTensor)
    # target_reward = np.array(list(d[1] for d in test_data))
    # pred_reward = nn_model(input_board)
    # pred_reward = pred_reward.detach().numpy()
    # loss = (np.square(pred_reward - target_reward)).mean()
    # print(loss)


