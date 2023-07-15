import torch
from data_handler import Buffer, load_data_and_train, BoardDataLoader, np_board_to_tensor_batch, load_data
from model import SmallNet, train, load_model_from_file, save_model, get_param
from self_play import GameMaster, play_eval
import time
from copy import deepcopy
# from pathos.helpers import mp
from torch import multiprocessing as mp
import json

if __name__ == "__main__":
    PATH = "training_data/"
    SUB_PATH = "Test7x7/"

    MODEL_NAME = "ValueNet_model"
    ModelClass = SmallNet
    training_pass = 80
    checkpoint_interval = 5
    MAX_TRAINING_DATA_SIZE = 100000
    transform_variance_count = 1
    num_workers = 60
    mp.set_start_method("spawn")
    mp.set_sharing_strategy('file_system')

    TRAINING_BATCH_SIZE = 32768
    batch_size = 32
    epoch = 1
    learning_rate = 0.005
    weight_decay = 0.001
    persistent_optimizer = True
    warm_up = False
    swap_size = False
    self_play_mode = True
    play_mode = "alpha_zero_reward"

    lr_decay = 0.5
    # lr_schedule = [30, 50, 70]
    lr_schedule = []

    dim = 7
    count = 5
    play_count = num_workers * 6
    self_play_eval_play_count = num_workers * 2
    n_sim = 400
    min_visit = 1
    AI_move_range = 1
    random_threshold = 6
    uct_temperature = 0
    rollout_weight = 0.25

    MODEL_NAME += "_" + str(n_sim) + "_" + str(play_count)
    nn_model = SmallNet()
    # NN_NAME = "ValueNet_model_400_180_pass100.pt"
    # print("PRELOADING", NN_NAME)
    # NN_PATH = PATH + SUB_PATH + NN_NAME
    # nn_model = load_model_from_file(NN_PATH, SmallNet)

    dir_path = PATH + SUB_PATH + "warm_up_1000" + ".txt"

    data = None
    if warm_up:
        print("WARMING-UP MODEL")
        t = time.time()
        data = load_data_and_train(dir_path, nn_model, data_count=10000, num_workers=0,
                                   lr=0.01, batch_size=batch_size, total_epoch=40, weight_decay=weight_decay,
                                   transform_variance_count=transform_variance_count)
        print("Loading data and training takes", time.time() - t)
        print(len(data))

    traindata = data if data is not None else BoardDataLoader(list(), MAX_TRAINING_DATA_SIZE, transform_variance_count)

    output_path = ""
    start_t = time.time()
    print("Beginning Self-play Training for", training_pass, "passes")
    print("MAX_TRAINING_DATA_SIZE", MAX_TRAINING_DATA_SIZE)
    print("TRAINING_BATCH_SIZE", TRAINING_BATCH_SIZE)
    print("TRANSFORM_VARIANCE_COUNT", transform_variance_count)
    print("UCT_TEMPERATURE", uct_temperature)
    print("ROLLOUT_WEIGHT", rollout_weight)
    print("SELF-PLAY", self_play_mode)
    print("PLAY MODE", play_mode)
    print()
    start_pass = 0
    optimizer = None
    # self_play_eval_results = {"prev_eval": list(), "prev_10_eval": list()}
    # param_buffer = Buffer(11)
    param = get_param(nn_model)

    for i in range(start_pass, start_pass + training_pass):
        print("PASS", i)
        if not persistent_optimizer:
            optimizer = None
            print("USING NEW OPTIMIZER")
        else:
            print("USING PERSISTENT OPTIMIZER")
        if i in lr_schedule:
            learning_rate = learning_rate * lr_decay
        learning_rate = optimizer.param_groups[0]['lr'] if optimizer is not None else learning_rate
        print("LEARNING RATE", learning_rate)
        # dir_path = PATH + SUB_PATH + "pass" + str(i+1)
        dir_path = None

        param = get_param(nn_model)
        AI1_params = {"n_sim": 400,
                      "min_visit": 1,
                      "AI_move_range": AI_move_range,
                      "mode": play_mode,
                      "random_threshold": random_threshold,
                      "model_param": deepcopy(param),
                      "uct_temperature": uct_temperature,
                      "rollout_weight": rollout_weight,
                      "random_transform": False,
                      "eval": None}
        # AI2_params = {"n_sim": 400,
        #               "min_visit": 1,
        #               "AI_move_range": AI_move_range,
        #               "mode": self_play_mode,
        #               "random_threshold": random_threshold,
        #               "model_param": deepcopy(param),
        #               "eval": None}

        game_master = GameMaster(dim, count, AI1_params, AI2_params=None, self_play_mode=self_play_mode,
                                 num_workers=num_workers,
                                 verbose=False, reward_outcome=True)
        t = time.time()
        game_master.start(play_count)
        new_data = game_master.data_points
        print("Time for " + str(play_count) + " games of self-play:", time.time() - t)
        np_board_to_tensor_batch(new_data, unsqueeze=False)
        print("Generated", len(new_data), "data points.")
        traindata.extend(new_data)

        output_path = PATH + SUB_PATH + MODEL_NAME + "_pass" + str(i) + ".pt"
        print("Total data count:", traindata.data_count())
        traindata.setup_training_batch(TRAINING_BATCH_SIZE)
        print("Training with " + str(len(traindata)) + " ")
        optimizer = train(nn_model, traindata, optimizer=optimizer, lr=learning_rate, weight_decay=weight_decay,
                          batch_size=batch_size, total_epoch=epoch, num_workers=0)

        # param = get_param(nn_model)
        # param_buffer.append(param)
        # print()
        # # Self-play Eval
        # if len(param_buffer) >= 2:
        #     print(f"Self-play eval pass {i} against pass {i-1}")
        #     param1, param2 = param_buffer[-1], param_buffer[-2]
        #     r = self_play_eval(dim, count, self_play_eval_play_count, param1, param2, num_workers=num_workers)
        #     self_play_eval_results["prev_eval"].append(r)
        # if len(param_buffer) >= param_buffer.capacity:
        #     print(f"Self-play eval pass {i} against pass {i - param_buffer.capacity + 1}")
        #     param1, param2 = param_buffer[-1], param_buffer[0]
        #     r = self_play_eval(dim, count, self_play_eval_play_count, param1, param2, num_workers=num_workers)
        #     self_play_eval_results["prev_10_eval"].append(r)

        if i % checkpoint_interval == 0:
            torch.save(nn_model.state_dict(), output_path)
        print("\n---------------------------\n")

    torch.save(nn_model.state_dict(), output_path)
    # with open(PATH + SUB_PATH + "self_play_eval.json", "w") as outfile:
    #     json.dump(self_play_eval_results, outfile)
    print("Finished Training", training_pass, "passes in", time.time()-start_t)

    # output_path = PATH + SUB_PATH + MODEL_NAME + "_final.pt"
    # print("\nFinal training with " + str(len(traindata)))
    # train(nn_model, traindata, lr=learning_rate/5, batch_size=batch_size,
    #       total_epoch=100, weight_decay=weight_decay, num_workers=0)
    # torch.save(nn_model.state_dict(), output_path)


