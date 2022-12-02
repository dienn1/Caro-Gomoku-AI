import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader
from model import Net, SmallNet, train, load_model
from self_play import self_play, get_evaluate_function
from data_handler import load_model_and_train, load_data_and_train


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
    SUB_PATH = "TicTacTest/"
    # NN_NAME = "mcts_nn_model_20000_20_20.pt"
    # NN_PATH = PATH + SUB_PATH + NN_NAME
    # nn_model = load_model(NN_PATH)
    # nn_model, loaded_data = load_model_and_train(parent_dir=PATH+SUB_PATH, model_name=NN_NAME, first=7, last=11,
    #                                              lr=0.0001, batch_size=64, total_epoch=100)
    # evaluate = get_evaluate_function(nn_model)

    dir_path = PATH + SUB_PATH + "pass0" + ".txt"
    # outfile = open(dir_path, "a")
    outfile = None
    model = SmallNet()
    data = load_data_and_train(dir_path, model, data_count=10000, total_epoch=25, no_duplicate=True)
    print(len(data))
    evaluate = get_evaluate_function(model)

    dim = 3
    count = 3
    play_count = 10
    n_sim1 = 0
    min_visit1 = 0
    n_sim2 = 1
    min_visit2 = 1
    mode1 = "greedy_post"
    mode2 = "random"

    eval1 = evaluate
    eval2 = None
    try:
        with torch.inference_mode():
            self_play(dim, count, play_count, n_sim1, min_visit1, eval1, mode1, n_sim2, min_visit2, eval2, mode2,
                      verbose=True, eval_model=evaluate, outfile=outfile)
    finally:
        # outfile.close()
        pass

