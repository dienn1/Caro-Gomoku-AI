import numpy as np
import torch
from torch.utils.data import Dataset
from model import Net, train, load_model
import time


class BoardDataLoader(Dataset):
    def __init__(self, data, max_recall):   # data format: [np_board, reward]
        if len(data) < max_recall:
            self.data = data
        else:
            self.data = data[-max_recall::1]

    def __len__(self):
        return len(self.data)

    # Return data point with format [board, reward]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            return self.data[idx][0], self.data[idx][1]
        sample = list()
        for i in idx:
            sample.append([self.data[i][0], self.data[i][1]])
        return sample


def load_raw_board_data(data_path):
    f = open(data_path, "r")
    data_dict = list()
    while True:
        line = f.readline()
        # when at the end of file or there's number (meaning reached the data count)
        if not line or line[:-1].isnumeric():
            break
        data = {"board": list(), "reward": 0}
        board = list()
        dim = len(line.split())
        for i in range(dim):
            row = [int(s) for s in line.split()]
            board.append(row)
            line = f.readline()
        data["board"] = board
        data["reward"] = float(line)
        data_dict.append(data)
        f.readline()    # newline here
    f.close()
    return data_dict


# Transform board into numpy array of (2, dim, dim)
# Used for processed board
def board_to_np(board_array, dim=15):
    np_board = np.zeros((2, 16, 16))
    for i in range(dim):
        for j in range(dim):
            pixel = board_array[i][j]
            if pixel > 0:
                np_board[pixel - 1, i, j] = 1
    return np_board


# return DATA FORMAT [transformed_tensor_board, reward]
def raw_data_transform(data_dict):
    transformed_data = list()
    dim = len(data_dict[0]["board"])
    for d in data_dict:
        np_board = board_to_np(d["board"], dim)
        transformed_data.append([np_board, d["reward"]])
    np_board_to_tensor_batch(transformed_data, unsqueeze=False)
    return transformed_data


def np_board_to_tensor(np_board, unsqueeze=False):
    board = torch.from_numpy(np_board).type(torch.FloatTensor)
    return torch.unsqueeze(board, 0) if unsqueeze else board


def np_board_to_tensor_batch(np_data, unsqueeze=False):
    for i in range(len(np_data)):
        np_data[i][0] = np_board_to_tensor(np_data[i][0], unsqueeze=unsqueeze)


def save_raw_data(f, mcts_ai, caro_board):
    reward = mcts_ai.predicted_reward()
    res = ""
    board_array = caro_board.get_board()
    dim = caro_board.get_dim()
    for i in range(dim):
        for j in range(dim):
            tmp = board_array[i][j]
            if tmp < 0:
                tmp = 2
            res += str(tmp) + " "
        res += "\n"
    res += str(reward) + "\n\n"
    f.write(res)


def process_board(board_array, dim=15):
    res = list()
    for i in range(dim):
        row = list()
        for j in range(dim):
            tmp = board_array[i][j]
            if tmp < 0:
                tmp = 2
            row.append(tmp)
        res.append(row)
    return res


# create a datapoint for NN training purpose
# FORMAT [board_array, reward]
def create_data_point(mcts_ai, caro_board):
    board_array = caro_board.get_board()
    dim = caro_board.get_dim()
    board_array = process_board(board_array, dim)
    np_board = board_to_np(board_array)
    return [np_board, mcts_ai.predicted_reward()]


if __name__ == "__main__":
    # data_dir = "training_data/attempt3/pass5.txt"
    # board_data = load_raw_board_data(data_dir)
    # transformed_board_data = raw_data_transform(board_data)
    # print(transformed_board_data[-2])
    # nn_model = Net()
    # PATH = "training_data/"
    # SUB_PATH = "attempt3/"
    # NN_NAME = "mcts_nn_model_20000_20_20.pt"
    # NN_PATH = PATH + SUB_PATH + NN_NAME
    # nn_model = load_model(NN_PATH)
    # traindata = BoardDataLoader(transformed_board_data, 49)
    # for i, data in enumerate(dataloader, 0):
    #     inputs, labels = data
    #     print(labels.shape)

    # nn_model(traindata[2][0].unsqueeze(0))
    # nn_model(traindata[1][0].unsqueeze(0))

    # t = time.time()
    # train(nn_model, traindata, lr=1e-3, batch_size=2, total_epoch=1)
    # print(time.time() - t)
    pass
