import numpy as np
import random
import torch
from torch.utils.data import Dataset
from model import train, load_model_from_file


class Buffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = list()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def append(self, d):
        self.data.append(d)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity::1]

    def extend(self, data):
        self.data.extend(data)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity::1]

    def __repr__(self):
        return str(self.data) + " | Capacity = " + str(self.capacity)

    def __str__(self):
        return str(self.data)


class BoardDataLoader(Dataset):
    def __init__(self, data, max_recall, transform_variance_count=2):   # data format: [np_board, reward]
        self.BOARD_ROTATIONS = np.array([-1, 0, 1, 2])
        self.BOARD_MIRROR = np.array([0, 1, 2, 3])
        self.BOARD_TRANSPOSE = np.array([0, 1])
        self.BOARD_TRANSFORMS = list()      # store possible transforms in (rotation, mirror, transpose) tuple
        for b_rot in self.BOARD_ROTATIONS:
            for b_mir in self.BOARD_MIRROR:
                for b_t in self.BOARD_TRANSPOSE:
                    self.BOARD_TRANSFORMS.append((b_rot, b_mir, b_t))
        self.transform_variance_count = transform_variance_count
        self.max_recall = max_recall
        if len(data) < self.max_recall:
            self.data = data
        else:
            self.data = data[-self.max_recall::1]
        self.training_batch = list()

    @staticmethod
    def hash_tensor(tensor):
        t = torch.flatten(tensor)
        return tuple(i.item() for i in t)

    def __len__(self):
        return len(self.training_batch)

    def data_count(self):
        return len(self.data)

    # initialize new training batch
    def setup_training_batch(self, training_batch_size):
        training_batch_size = min(training_batch_size, self.data_count())
        self.training_batch = list()
        data_ind = np.random.choice(self.data_count(), size=training_batch_size, replace=False)
        for i in data_ind:
            self.training_batch.append(self.data[i])

    def append(self, d):
        self.data.append(d)
        if len(self.data) > self.max_recall:
            self.data = self.data[-self.max_recall::1]

    def extend(self, data):
        self.data.extend(data)
        if len(self.data) > self.max_recall:
            self.data = self.data[-self.max_recall::1]

    # Return data point with format [board, reward]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            board = self.transform_board(idx, count=self.transform_variance_count)
            return board, self.data[idx][1]
        sample = list()
        for i in idx:
            sample.append([self.data[i][0], self.data[i][1]])
        return sample

    # randomly rotate, mirror, transpose the board index idx
    def transform_board(self, idx, count=2):
        transforms = random.sample(self.BOARD_TRANSFORMS, count)
        transformed_boards = list()
        for i in range(count):
            board = self.data[idx][0]
            rotate, mirror, transpose = transforms[i][0], transforms[i][1], transforms[i][2]
            if rotate != 0:
                board = torch.rot90(board, k=rotate, dims=(1, 2))
            if mirror != 0:
                if mirror == 3:
                    board = torch.flip(board, (1, 2))
                else:
                    board = torch.flip(board, (mirror,))
            if transpose != 0:
                board = board.transpose(1, 2)
            transformed_boards.append(board)
        return transformed_boards

    def mean(self):
        return np.mean(list(self.data[i][1] for i in range(len(self.data))))

    def result_distribution(self):
        res_dict = {-1: 0, 0: 0, 1: 0}
        for i in range(len(self.data)):
            res = self.data[i][1]
            if res in res_dict:
                res_dict[res] += 1
        return res_dict


# Load raw data from file
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


# Convert raw_board_array (-1:O, 1:X) to the correct dim board_array with (1:X, 2:O)
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


# Transform np_board to processed board
def np_to_board(np_board):
    dim = np_board.shape[1]
    board_array = np.zeros((dim, dim))
    for n in range(2):
        for i in range(dim):
            for j in range(dim):
                board_array[i, j] += np_board[n, i, j] * (n + 1)
    return board_array


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


# Save raw board (-1:O, 1:X) with reward
def save_raw_board(f, reward, raw_board_array, dim):
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


# Save np_board (2, dim, dim) with reward
def save_np_board(f, reward, np_board):
    res = ""
    processed_board = np_to_board(np_board)
    dim = processed_board.shape[0]
    for i in range(dim):
        for j in range(dim):
            res += str(int(processed_board[i][j])) + " "
        res += "\n"
    res += str(reward) + "\n\n"
    f.write(res)


# create a datapoint for NN training purpose
# FORMAT [board_array, reward]
def create_data_point(reward, board_array, dim):
    board_array = process_board(board_array, dim)
    np_board = board_to_np(board_array, dim)
    return [np_board, reward]


def load_data(data_dir, data_count=10000, transform_variance_count=2):
    board_data = load_raw_board_data(data_dir)
    transformed_board_data = raw_data_transform(board_data)
    traindata = BoardDataLoader(transformed_board_data, data_count, transform_variance_count=transform_variance_count)
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
    traindata = BoardDataLoader(data, data_count)
    return traindata


def load_data_and_train(data_dir, model, data_count=10000, training_batch_size=1024,
                        lr=0.0001, weight_decay=0, batch_size=32, total_epoch=5, transform_variance_count=2,
                        num_workers=0):
    traindata = load_data(data_dir, data_count=data_count, transform_variance_count=transform_variance_count)
    traindata.setup_training_batch(training_batch_size)
    train(model, traindata, total_epoch=total_epoch, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
          num_workers=num_workers)
    return traindata


def batch_load_data_and_train(parent_dir, model, first, last=None, data_count=10000,
                              lr=0.0001, weight_decay=0, batch_size=64, total_epoch=100):
    # if last is None:
    #     last = first
    # data = list()
    # for i in range(first, last + 1):
    #     data_dir = parent_dir + "pass" + str(i) + ".txt"
    #     board_data = load_raw_board_data(data_dir)
    #     transformed_board_data = raw_data_transform(board_data)
    #     data.extend(transformed_board_data)
    traindata = batch_load_data(parent_dir, first, last, data_count)
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

    # b = Buffer(11)
    # print(b)
    # for i in range(0, 20):
    #     b.append(i)
    #     if len(b) >= 2:
    #         print(b[-1], b[-2])
    #     if len(b) >= 10:
    #         print(b[-1], b[0])
    #     print(b)
    pass
