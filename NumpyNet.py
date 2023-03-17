import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class NPLinear:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, x):
        return np.matmul(x, self.weights.T) + self.bias

    def __call__(self, x):
        return self.forward(x)


class NPConv:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.num_layers = self.weights.shape[0]
        self.in_channels = self.weights.shape[1]
        self.kernel_size = self.weights.shape[2]

    def forward(self, x):
        out = list()
        out_dim = x.shape[-1] - self.kernel_size + 1
        for i in range(self.num_layers):
            layer = np.zeros((out_dim, out_dim))
            for x1 in range(out_dim):
                next_x1 = x1 + self.kernel_size
                for x2 in range(out_dim):
                    next_x2 = x2 + self.kernel_size
                    # convolve
                    for j in range(self.in_channels):
                        layer[x1, x2] += np.sum(self.weights[i, j] * x[j, x1:next_x1, x2:next_x2])
                    layer[x1, x2] += self.bias[i]
            out.append(layer)
        return np.array(out)

    def __call__(self, x):
        return self.forward(x)


def np_relu(x):
    return x * (x > 0)


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 5, 3)
        self.fc1 = nn.Linear(125, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x.flatten()))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    conv1_weights = -np.ones((5, 2, 3, 3))
    conv1_bias = np.arange(5)
    for i in range(5):
        conv1_bias[i] = (i + 1) ** 3

    fc1_weights = np.ones((50, 125))
    fc1_bias = np.arange(50)
    for i in range(50):
        fc1_bias[i] = (i + 1) ** 3

    fc2_weights = np.ones((1, 50))
    fc2_bias = np.arange(1)
    for i in range(1):
        fc2_bias[i] = (i + 1) ** 3

    example = np.zeros((2, 7, 7))
    for i in range(2):
        for j in range(7):
            for k in range(7):
                example[i, j, k] = (i + 1)**2 * j * k

    # torch_example = torch.rand(2, 7, 7)
    # example = torch_example.detach().numpy()
    #
    # model = TestNet()
    # res = model(torch.from_numpy(example))
    # param = list()
    # for n, p in model.named_parameters():
    #     print(n)
    #     t = p.detach().numpy()
    #     # print(t)
    #     print(t.shape)
    #     print()
    #     param.append(t)
    #
    # conv1_weights, conv1_bias = param[0], param[1]
    # fc1_weights, fc1_bias = param[2], param[3]
    # fc2_weights, fc2_bias = param[4], param[5]

    print(example)
    conv1 = NPConv(conv1_weights, conv1_bias)
    fc1 = NPLinear(fc1_weights, fc1_bias)
    fc2 = NPLinear(fc2_weights, fc2_bias)

    example = np_relu(conv1(example))
    example = np_relu(fc1(example.flatten()))
    example = fc2(example)

    print(example)
    # print(res)
