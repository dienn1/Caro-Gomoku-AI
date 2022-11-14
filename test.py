import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader
from model import Net, train, load_model
from self_play import self_play, get_evaluate_function
from main import load_model_and_train


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
    PATH = "training_data/"
    SUB_PATH = "attempt3/"
    NN_NAME = "mcts_nn_model_20_20000.pt"
    NN_PATH = PATH + SUB_PATH + NN_NAME
    nn_model = load_model(NN_PATH)
    # nn_model, loaded_data = load_model_and_train(parent_dir=PATH+SUB_PATH, model_name=NN_NAME, first=7, last=11,
    #                                              lr=0.0001, batch_size=64, total_epoch=100)
    evaluate = get_evaluate_function(nn_model)

    dim = 15
    count = 20
    n_sim1 = 20000
    min_visit1 = 20
    n_sim2 = 20000
    min_visit2 = 20

    eval1 = evaluate
    eval2 = None
    with torch.inference_mode():
        self_play(dim, count, n_sim1, min_visit1, eval1, n_sim2, min_visit2, eval2, verbose=True, eval_model=evaluate)
    # outfile.close()

