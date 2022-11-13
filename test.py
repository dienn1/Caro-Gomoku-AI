import torch
from data_handler import load_raw_board_data, board_to_np, raw_data_transform, np_board_to_tensor, BoardDataLoader
from model import Net, train, load_model
from self_play import self_play
from main import load_model_and_train

if __name__ == "__main__":
    # a = [i for i in range(50)]
    # print(a[-1:-10:-1])
    # print(a[-10::1])
    # PATH = "training_data/"
    # SUB_PATH = "attempt1/"
    # NN_NAME = "mcts_nn_model_20000.pt"
    # NN_PATH = PATH + SUB_PATH + NN_NAME
    # # nn_model = load_model(NN_PATH)
    # nn_model, loaded_data = load_model_and_train(parent_dir=PATH+SUB_PATH, model_name=NN_NAME, first=7, last=10,
    #                                              lr=0.0001, batch_size=64, total_epoch=100)
    # def evaluate(board, player):
    #     player = 2 if player < 0 else player
    #     board = board_to_np(board, player)
    #     board = np_board_to_tensor(board, unsqueeze=True)
    #     return nn_model(board)

    dim = 15
    count = 10
    n_sim1 = 20000
    min_visit1 = 20
    n_sim2 = 20000
    min_visit2 = 20

    eval1 = None
    eval2 = None
    with torch.no_grad():
        self_play(dim, count, n_sim1, min_visit1, eval1, n_sim2, min_visit2, eval2, verbose=True, eval_model=None)
