import torch
from data_handler import load_raw_board_data, load_data_and_train, BoardDataLoader, np_board_to_tensor_batch
from model import Net, SmallNet, train, load_model, save_model
from self_play import self_play, get_evaluate_function, SelfPlay
import time

if __name__ == "__main__":
    PATH = "training_data/"
    SUB_PATH = "Test7x7/"

    MODEL_NAME = "mcts_nn_model"
    training_pass = 20
    MAX_TRAINING_DATA_SIZE = 1000000

    batch_size = 32
    training_epoch = 2
    learning_rate = 0.0005
    warm_up = True

    dim = 7
    count = 5
    play_count = 50
    n_sim = 400
    min_visit = 1
    AI_move_range = 1

    MODEL_NAME += "_" + str(n_sim) + "_" + str(min_visit) + "_" + str(play_count)
    nn_model = SmallNet()

    dir_path = PATH + SUB_PATH + "pass0" + ".txt"

    if warm_up:
        print("WARMING-UP MODEL")
        t = time.time()
        data = load_data_and_train(dir_path, nn_model, data_count=100000,
                                   lr=0.0005, batch_size=64, total_epoch=5, no_duplicate=True)
        print("Loading data and training takes", time.time() - t)
        print(len(data))
    evaluate = get_evaluate_function(nn_model)

    traindata = BoardDataLoader(list(), MAX_TRAINING_DATA_SIZE, no_duplicate=True)

    try:
        for i in range(training_pass):
            dir_path = PATH + SUB_PATH + "pass" + str(i+1) + ".txt"
            outfile = open(dir_path, "a")

            AI1_params = {"n_sim": 400,
                          "min_visit": 1,
                          "AI_move_range": AI_move_range,
                          "mode": "alpha_zero",
                          "eval": evaluate}
            AI2_params = {"n_sim": 400,
                          "min_visit": 1,
                          "AI_move_range": AI_move_range,
                          "mode": "alpha_zero",
                          "eval": evaluate}
            game_master = SelfPlay(dim, count, AI1_params, AI2_params, eval_model=evaluate,
                                   loss=None, verbose=False, outfile=outfile, reward_outcome=True)
            t = time.time()
            with torch.inference_mode():
                new_data = game_master.self_play(play_count)
            print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
            outfile.close()

            np_board_to_tensor_batch(new_data, unsqueeze=False)
            traindata.extend(new_data)
            print("Training with " + str(len(traindata)) + " ")
            train(nn_model, traindata, lr=learning_rate, batch_size=batch_size, total_epoch=training_epoch)
            torch.save(nn_model.state_dict(), PATH + SUB_PATH + MODEL_NAME + "_temp" + str(i) + ".pt")

            evaluate = get_evaluate_function(nn_model)
    finally:
        torch.save(nn_model.state_dict(), PATH + SUB_PATH + MODEL_NAME + "_interrupted.pt")
