import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader
from model import Net, train, load_model
from self_play import self_play

if __name__ == "__main__":
    # a = [i for i in range(50)]
    # print(a[-1:-10:-1])
    # print(a[-10::1])
    NN_PATH = "mcts_nn_model_20000.pt"
    model = load_model(NN_PATH)

    dim = 15
    count = 10
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
    eval2 = None

    self_play(dim, count, n_sim1, min_visit1, eval1, n_sim2, min_visit2, eval2, verbose=True)