import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader
from model import Net, train
from self_play import self_play

CHAR = {-1: "O", 0: ".", 1: "X"}


def str_board(_board):
    res = ""
    for i in range(dim):
        row_str = ""
        for j in range(dim):
            row_str += CHAR[_board[i][j]] + " "
        res += row_str + '\n'
    return res


def load_data_and_train(data_dir, model):
    board_data = load_raw_board_data(data_dir, 15)
    transformed_board_data = raw_data_transform(board_data)
    traindata = BoardDataLoader(transformed_board_data, 1000)
    train(model, traindata)
    return transformed_board_data


if __name__ == "__main__":
    PATH = "training_data/"
    MODEL_NAME = "mcts_nn_model_20000"
    training_pass = 10
    MAX_TRAINING_DATA_SIZE = 1000

    dim = 15
    count = 20
    n_sim1 = 20000
    min_visit1 = 20
    n_sim2 = 20000
    min_visit2 = 20

    nn_model = Net()

    def evaluate(board, player):
        player = 2 if player < 0 else player
        board = board_to_np(board, player)
        board = np_board_to_tensor(board, unsqueeze=True)
        return nn_model(board)
    eval1 = evaluate
    eval2 = evaluate

    data = load_data_and_train("training_data/pass0.txt", nn_model)
    try:
        for i in range(training_pass):
            dir_path = "training_data/pass" + str(i+1) + ".txt"
            outfile = open(dir_path, "a")
            with torch.no_grad():
                new_data = self_play(dim, count, n_sim1, min_visit1, eval1, n_sim2, min_visit2, eval2, verbose=False, outfile=outfile)
                data.extend(new_data)
            outfile.close()
            traindata = BoardDataLoader(data, MAX_TRAINING_DATA_SIZE)
            train(nn_model, traindata)
            torch.save(nn_model.state_dict(), PATH + MODEL_NAME + "_temp" + str(i) + ".pt")
    finally:
        torch.save(nn_model.state_dict(), PATH + MODEL_NAME + ".pt")
