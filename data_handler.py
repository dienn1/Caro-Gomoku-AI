import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import Net, train
import time
from copy import copy, deepcopy


class BoardDataLoader(Dataset):
    def __init__(self, data, max_recall):   # data format: [np_board, winrate]
        if len(data) < max_recall:
            self.data = data
        else:
            self.data = data[-max_recall::1]

    def __len__(self):
        return len(self.data)

    # Return data point with format [board, winrate]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            return self.data[idx][0], self.data[idx][1]
        sample = list()
        for i in idx:
            sample.append([self.data[i][0], self.data[i][1]])
        return sample


def load_raw_board_data(data_path, dim):
    f = open(data_path, "r")
    data_dict = list()
    while True:
        line = f.readline()
        if not line or int(line) > 2:   # when at the end of file or the number is > 2 (meaning reached the data count)
            break
        data = {"player": int(line), "board": list(), "winrate": 0}
        board = list()
        for i in range(dim):
            line = f.readline()
            row = [int(s) for s in line.split()]
            board.append(row)
        data["board"] = board
        line = f.readline()
        data["winrate"] = float(line)
        data_dict.append(data)
        f.readline()    # newline here
    return data_dict


def reverse(n):
    return 2 if n == 1 else 1


# Transform board into numpy array of (2, dim, dim), X and O will be swapped if the player is 2 (O)
# Used for processed board
def board_to_np(board_array, player, dim=15):
    np_board = np.zeros((2, 16, 16))
    for i in range(dim):
        for j in range(dim):
            pixel = board_array[i][j]
            if pixel > 0:
                pixel = pixel if player == 1 else reverse(pixel)
                np_board[pixel - 1, i, j] = 1
    return np_board


# return DATA FORMAT [transformed_tensor_board, winrate]
def raw_data_transform(data_dict):
    transformed_data = list()
    dim = len(data_dict[0]["board"])
    for d in data_dict:
        np_board = board_to_np(d["board"], d["player"], dim)
        transformed_data.append([np_board, d["winrate"]])
    np_board_to_tensor_batch(transformed_data, unsqueeze=False)
    return transformed_data


def np_board_to_tensor(np_board, unsqueeze=False):
    board = torch.from_numpy(np_board).type(torch.FloatTensor)
    return torch.unsqueeze(board, 0) if unsqueeze else board


def np_board_to_tensor_batch(np_data, unsqueeze=False):
    for i in range(len(np_data)):
        np_data[i][0] = np_board_to_tensor(np_data[i][0], unsqueeze=unsqueeze)


def save_raw_data(f, mcts_ai, caro_board):
    winrate = mcts_ai.predicted_winrate()
    player = mcts_ai.get_player()
    if player < 0:
        player = 2
    res = str(player) + "\n"
    board_array = caro_board.get_board()
    dim = caro_board.get_dim()
    for i in range(dim):
        for j in range(dim):
            tmp = board_array[i][j]
            if tmp < 0:
                tmp = 2
            res += str(tmp) + " "
        res += "\n"
    res += str(winrate) + "\n\n"
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
# FORMAT [board_array, winrate]
def create_data_point(mcts_ai, caro_board):
    board_array = caro_board.get_board()
    dim = caro_board.get_dim()
    board_array = process_board(board_array, dim)
    np_board = board_to_np(board_array, mcts_ai.get_player())
    return [np_board, mcts_ai.predicted_winrate()]


if __name__ == "__main__":
    data_dir = "training_data/pass0.txt"
    board_data = load_raw_board_data(data_dir, 15)
    transformed_board_data = raw_data_transform(board_data)
    # print(transformed_board_data[-2])
    net = Net()
    traindata = BoardDataLoader(transformed_board_data, 1000)
    # dataloader = DataLoader(traindata, batch_size=4, shuffle=True, num_workers=0)
    # for i, data in enumerate(dataloader, 0):
    #     inputs, labels = data
    #     print(labels.shape)
    t = time.time()
    train(net, traindata, lr=1e-3, batch_size=16)
    print(time.time() - t)

