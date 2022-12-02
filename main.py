import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader, np_board_to_tensor_batch
from model import Net, train, load_model, save_model
from self_play import self_play, get_evaluate_function


if __name__ == "__main__":
    PATH = "training_data/attemptTicTac/"
    MODEL_NAME = "mcts_nn_model"
    training_pass = 11
    MAX_TRAINING_DATA_SIZE = 1000

    batch_size = 64
    training_epoch = 25
    learning_rate = 0.0001

    dim = 15
    count = 20
    n_sim = 20000
    min_visit = 20

    MODEL_NAME += "_" + str(n_sim) + "_" + str(min_visit) + "_" + str(count)

    nn_model = Net()

    evaluate = get_evaluate_function(nn_model)
    eval1 = evaluate
    eval2 = evaluate

    traindata = BoardDataLoader(list(), MAX_TRAINING_DATA_SIZE, no_duplicate=True)
    # data = load_data_and_train("training_data/pass0.txt", nn_model)     # warm-up the model
    try:
        for i in range(training_pass):
            dir_path = PATH + "pass" + str(i+1) + ".txt"
            outfile = open(dir_path, "a")
            with torch.inference_mode():
                new_data = self_play(dim, count, n_sim, min_visit, eval1, n_sim, min_visit, eval2, verbose=False, outfile=outfile)
            np_board_to_tensor_batch(new_data, unsqueeze=False)
            traindata.extend(new_data)
            outfile.close()
            train(nn_model, traindata, lr=learning_rate, batch_size=batch_size, total_epoch=training_epoch)
            torch.save(nn_model.state_dict(), PATH + MODEL_NAME + "_temp" + str(i) + ".pt")
    finally:
        torch.save(nn_model.state_dict(), PATH + MODEL_NAME + ".pt")
