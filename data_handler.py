import numpy as np
import torch
from torch.utils.data import Dataset
from model import train, load_model_from_file


class BoardDataLoader(Dataset):
    def __init__(self, data, max_recall, no_duplicate=False):   # data format: [np_board, reward]
        self.max_recall = max_recall    # max_recall only applies if no_duplicate=False
        if len(data) < self.max_recall:
            self.data = data
        else:
            self.data = data[-self.max_recall::1]
        self.no_duplicate = no_duplicate
        if self.no_duplicate:
            self.unique_data = dict()
            filtered_data = list()
            for d in self.data:
                hashed_d = BoardDataLoader.hash_tensor(d[0])
                if hashed_d not in self.unique_data:
                    self.unique_data[hashed_d] = {"reward": d[1], "count": 1, "index": 0, "tensor": d[0]}
                else:
                    # incremental average
                    self.unique_data[hashed_d]["count"] += 1
                    current_avg = self.unique_data[hashed_d]["reward"]
                    self.unique_data[hashed_d]["reward"] = current_avg + (d[1] - current_avg)/self.unique_data[hashed_d]["count"]

            for i, hashed_d in enumerate(self.unique_data):
                filtered_data.append([self.unique_data[hashed_d]["tensor"], self.unique_data[hashed_d]["reward"]])
                self.unique_data[hashed_d]["index"] = i
            self.data = filtered_data

    @staticmethod
    def hash_tensor(tensor):
        t = torch.flatten(tensor)
        return tuple(i.item() for i in t)

    def __len__(self):
        return len(self.data)

    def append(self, d):
        if not self.no_duplicate:
            self.data.append(d)
            if len(self.data) > self.max_recall:    # max_recall only applies if no_duplicate=False
                self.data = self.data[-self.max_recall::1]
            return
        hashed_d = BoardDataLoader.hash_tensor(d[0])
        if hashed_d not in self.unique_data:
            self.unique_data[hashed_d] = {"reward": d[1], "count": 1, "index": len(self.data), "tensor": d[0]}
            self.data.append(d)
        else:
            # incremental average
            self.unique_data[hashed_d]["count"] += 1
            current_avg = self.unique_data[hashed_d]["reward"]
            self.unique_data[hashed_d]["reward"] = current_avg + (d[1] - current_avg) / self.unique_data[hashed_d]["count"]
            self.data[self.unique_data[hashed_d]["index"]] = [self.unique_data[hashed_d]["tensor"], self.unique_data[hashed_d]["reward"]]

    def extend(self, data):
        if not self.no_duplicate:   # max_recall only applies if no_duplicate=False
            self.data.extend(data)
            if len(self.data) > self.max_recall:
                self.data = self.data[-self.max_recall::1]
            return
        for d in data:
            self.append(d)

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

    def mean(self):
        return np.mean(list(self.data[i][1] for i in range(len(self.data))))

    def result_distribution(self):
        res_dict = {-1: 0, 0: 0, 1: 0}
        for i in range(len(self.data)):
            res = self.data[i][1]
            if res in res_dict:
                res_dict[res] += 1
        return res_dict


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
    np_board = np.zeros((2, dim, dim))
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


def save_raw_data(f, reward, raw_board_array, dim):
    res = ""
    for i in range(dim):
        for j in range(dim):
            tmp = raw_board_array[i][j]
            if tmp < 0:
                tmp = 2
            res += str(tmp) + " "
        res += "\n"
    res += str(reward) + "\n\n"
    f.write(res)


def process_board(raw_board_array, dim=15):
    res = list()
    for i in range(dim):
        row = list()
        for j in range(dim):
            tmp = raw_board_array[i][j]
            if tmp < 0:
                tmp = 2
            row.append(tmp)
        res.append(row)
    return res


# create a datapoint for NN training purpose
# FORMAT [board_array, reward]
def create_data_point(reward, board_array, dim):
    board_array = process_board(board_array, dim)
    np_board = board_to_np(board_array, dim)
    return [np_board, reward]


def load_data(data_dir, data_count=10000, no_duplicate=False):
    board_data = load_raw_board_data(data_dir)
    transformed_board_data = raw_data_transform(board_data)
    traindata = BoardDataLoader(transformed_board_data, data_count, no_duplicate=no_duplicate)
    return traindata


def batch_load_data(parent_dir, first, last=None, data_count=10000, no_duplicate=False):
    if last is None:
        last = first
    data = list()
    for i in range(first, last + 1):
        data_dir = parent_dir + "pass" + str(i) + ".txt"
        board_data = load_raw_board_data(data_dir)
        transformed_board_data = raw_data_transform(board_data)
        data.extend(transformed_board_data)
    traindata = BoardDataLoader(data, data_count, no_duplicate=no_duplicate)
    return traindata


def load_data_and_train(data_dir, model, data_count=10000, lr=0.0001, weight_decay=0,
                        batch_size=32, total_epoch=5, no_duplicate=False, num_workers=0):
    traindata = load_data(data_dir, data_count=data_count, no_duplicate=no_duplicate)
    train(model, traindata, total_epoch=total_epoch, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
          num_workers=num_workers)
    return traindata


def batch_load_data_and_train(parent_dir, model, first, last=None, data_count=10000, no_duplicate=False,
                              lr=0.0001, weight_decay=0, batch_size=64, total_epoch=100):
    # if last is None:
    #     last = first
    # data = list()
    # for i in range(first, last + 1):
    #     data_dir = parent_dir + "pass" + str(i) + ".txt"
    #     board_data = load_raw_board_data(data_dir)
    #     transformed_board_data = raw_data_transform(board_data)
    #     data.extend(transformed_board_data)
    traindata = batch_load_data(parent_dir, first, last, data_count, no_duplicate)
    train(model, traindata, lr=lr, batch_size=batch_size, total_epoch=total_epoch, weight_decay=weight_decay)
    return traindata


def load_model_and_train(parent_dir, model_name, first, last=None, lr=0.0001, batch_size=64, total_epoch=100):
    model = load_model_from_file(parent_dir + model_name)
    data = batch_load_data_and_train(parent_dir, model, first, last, lr=lr, batch_size=batch_size, total_epoch=total_epoch)
    return model, data


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
