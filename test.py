import torch
from data_handler import load_raw_board_data, load_data, batch_load_data, BoardDataLoader
from model import SmallNet, train, load_model_from_file, FFNet, optimize_model
from self_play import get_evaluate_function, SelfPlay, mse_loss
from data_handler import load_model_and_train, load_data_and_train
from NumpyNet import NPLinear, NPConv
import random
import time
import numpy as np
from Model_pybind import SmallNetTest
from copy import deepcopy, copy
# from pathos.helpers import mp
import torch.multiprocessing as mp


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
    # mp.set_start_method("spawn")

    dim = 7
    count = 5
    play_count = 128

    PATH = "training_data/"
    SUB_PATH = "Test7x7/"

    # data = batch_load_data(parent_dir=PATH+SUB_PATH, first=1, last=20, data_count=100000, no_duplicate=False)
    # print(len(data))
    # print(data.mean())
    # print(data.result_distribution())

    NN_NAME = "mcts_nn_model_400_50.pt"
    NN_PATH = PATH + SUB_PATH + NN_NAME
    # nn_model = load_model(NN_PATH, SmallNet)
    # nn_model, loaded_data = load_model_and_train(parent_dir=PATH+SUB_PATH, model_name=NN_NAME, first=7, last=11,
    #                                              lr=0.0001, batch_size=64, total_epoch=100)
    dir_path = PATH + SUB_PATH + "pass0.txt"
    # outfile_path = PATH + SUB_PATH + "mp_test"
    outfile_path = None
    model = SmallNet()
    # model = FFNet(7*7)
    t = time.time()
    data = load_data_and_train(dir_path, model, data_count=1000, num_workers=0,
                               lr=0.0001, batch_size=64, total_epoch=10, weight_decay=0.001, no_duplicate=False)
    print("Loading data and training takes", time.time()-t)
    print(len(data))
    del data

    param = list()
    for n, p in model.named_parameters():
        print(n)
        t = p.detach().numpy()
        # print(t)
        print(t.shape)
        print()
        param.append(t)

    np_conv = NPConv(param[0], param[1])
    # np_conv = NPConv(torch.rand(256, 2, 3, 3).detach().numpy(), torch.rand(256).detach().numpy())
    rand_input = torch.rand(2, 7, 7)
    rand_input_np = rand_input.detach().numpy()
    # a = model.conv1(rand_input).detach().numpy()
    # b = np_conv(rand_input_np)
    # print(np.sum(np.abs(a - b))/(256*5*5))
    print(len(param))

    conv1_weights, conv1_bias = param[0].astype("float"), param[1].astype("float")
    conv2_weights, conv2_bias = param[2].astype("float"), param[3].astype("float")
    fc1_weights, fc1_bias = param[4].astype("float"), param[5].astype("float")
    fc2_weights, fc2_bias = param[6].astype("float"), param[7].astype("float")
    fc3_weights, fc3_bias = param[8].astype("float"), param[9].astype("float")

    pybind_model_test = SmallNetTest(rand_input_np,
                                     conv1_weights, conv1_bias,
                                     conv2_weights, conv2_bias,
                                     fc1_weights, fc1_bias,
                                     fc2_weights, fc2_bias,
                                     fc3_weights, fc3_bias)
    print()
    res = model(torch.reshape(rand_input, (1, 2, 7, 7)))
    print(pybind_model_test, res[0, 0].item())

    # np_ff = [NPLinear(None, None)] * int(len(param)/2)
    # for i in range(int(len(param)/2)):
    #     np_linear = NPLinear(param[2*i], param[2*i + 1])
    #     np_ff[i] = np_linear
    #
    # rand_input = torch.rand(10, 98)
    # rand_input_np = rand_input.detach().numpy()
    # a = model(rand_input).detach().numpy()
    # b = rand_input_np
    # for i in range(len(np_ff)-1):
    #     b = np_ff[i](b)
    #     b = np.maximum(b, 0)    # Relu
    # b = np_ff[-1](b)
    # b = np.tanh(b)
    #
    # print(a)
    # print(b)
    # print(a-b)

    exit()

    AI_move_range = 1
    random_threshold = 4
    optimize = False
    # traced_model = optimize_model(model)
    traced_model = model

    # AI1_params = {"n_sim": 10000,
    #               "min_visit": 10,
    #               "AI_move_range": AI_move_range,
    #               "mode": "alpha_zero",
    #               "random_threshold": random_threshold,
    #               "model_state_dict": None,
    #               "eval": None}
    # AI2_params = {"n_sim": 10000,
    #               "min_visit": 10,
    #               "AI_move_range": AI_move_range,
    #               "mode": "alpha_zero",
    #               "random_threshold": random_threshold,
    #               "model_state_dict": None,
    #               "eval": None}

    AI1_params = {"n_sim": 400,
                  "min_visit": 1,
                  "AI_move_range": AI_move_range,
                  "mode": "alpha_zero",
                  "random_threshold": random_threshold,
                  "model_state_dict": traced_model.state_dict(),
                  "eval": None}
    AI2_params = {"n_sim": 400,
                  "min_visit": 1,
                  "AI_move_range": AI_move_range,
                  "mode": "alpha_zero",
                  "random_threshold": random_threshold,
                  "model_state_dict": traced_model.state_dict(),
                  "eval": None}

    game_master = SelfPlay(dim, count, AI1_params, AI2_params, num_workers=0, torch_optimize=optimize,
                           verbose=False, eval_model=None, outfile_path=outfile_path, reward_outcome=False)
    t = time.time()
    # data = game_master.start(play_count)
    with torch.inference_mode():
        data = game_master.start(play_count)
    print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
    print(len(data))

