import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader
from model import Net, SmallNet, train, load_model, FFNet
from self_play import self_play, get_evaluate_function, SelfPlay
from data_handler import load_model_and_train, load_data_and_train
import random
import time


def random_baseline(board, dim):
    return random.uniform(-1, 1)


def beta_baseline(board, dim):
    alpha, beta = 4, 4
    return 2*random.betavariate(alpha, beta) - 1


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
    dim = 7
    count = 5
    play_count = 10

    PATH = "training_data/"
    SUB_PATH = "Test7x7/"
    NN_NAME = "mcts_nn_model_400_1_50.pt"
    NN_PATH = PATH + SUB_PATH + NN_NAME
    nn_model = load_model(NN_PATH, SmallNet)
    # nn_model, loaded_data = load_model_and_train(parent_dir=PATH+SUB_PATH, model_name=NN_NAME, first=7, last=11,
    #                                              lr=0.0001, batch_size=64, total_epoch=100)
    # evaluate = get_evaluate_function(nn_model)

    dir_path = PATH + SUB_PATH + "pass0" + ".txt"
    # outfile = open(dir_path, "a")
    outfile = None
    # dir_path_t = PATH + SUB_PATH + "pass1" + ".txt"
    # outfile = open(dir_path_t, "a")
    # model = Net()
    model = SmallNet()
    # model = FFNet(dim*dim)
    t = time.time()
    data = load_data_and_train(dir_path, model, data_count=100000,
                               lr=0.0005, batch_size=64, total_epoch=5, no_duplicate=True)
    print("Loading data and training takes", time.time()-t)
    print(len(data))
    evaluate = get_evaluate_function(model)
    eval2 = get_evaluate_function(nn_model)
    # evaluate = beta_baseline
    # evaluate = None
    # exit()

    AI_move_range = 1
    AI1_params = {"n_sim": 1000,
                  "min_visit": 10,
                  "AI_move_range": AI_move_range,
                  "mode": "greedy_visit",
                  "eval": None}
    AI2_params = {"n_sim": 1000,
                  "min_visit": 10,
                  "AI_move_range": AI_move_range,
                  "mode": "greedy_visit",
                  "eval": None}

    game_master = SelfPlay(dim, count, AI1_params, AI2_params,
                           verbose=True, eval_model=eval2, outfile=outfile, reward_outcome=True)
    try:
        t = time.time()
        with torch.inference_mode():
            game_master.self_play(play_count)
        print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
    finally:
        if outfile:
            outfile.close()
        # pass

