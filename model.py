import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from Model_pybind import SmallNet as SmallNetPybind


def optimize_model(model, input_dim=7):
    traced_model = model
    torch.jit.enable_onednn_fusion(True)
    sample_input = torch.rand(32, 2, input_dim, input_dim)
    # Tracing the model with example input
    traced_model = torch.jit.trace(traced_model.eval(), sample_input)
    # Invoking torch.jit.freeze
    traced_model = torch.jit.freeze(traced_model)
    model.train()
    return traced_model


def initialize_SmallNetPybind(param):
    conv1_weights, conv1_bias = param[0].astype("float"), param[1].astype("float")
    conv2_weights, conv2_bias = param[2].astype("float"), param[3].astype("float")
    fc1_weights, fc1_bias = param[4].astype("float"), param[5].astype("float")
    fc2_weights, fc2_bias = param[6].astype("float"), param[7].astype("float")
    fc3_weights, fc3_bias = param[8].astype("float"), param[9].astype("float")

    model_pybind = SmallNetPybind(conv1_weights, conv1_bias,
                                  conv2_weights, conv2_bias,
                                  fc1_weights, fc1_bias,
                                  fc2_weights, fc2_bias,
                                  fc3_weights, fc3_bias)
    return model_pybind


# Make a param list for the nn model
def get_param(model):
    param = list()
    for n, p in model.named_parameters():
        t = p.detach().numpy()
        param.append(t)
    return param


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 256, 3)
        self.conv2 = nn.Conv2d(256, 16, 1)
        self.fc1 = nn.Linear(400, 800)
        self.fc2 = nn.Linear(800, 256)
        self.fc3 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        return x


class FFNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size*2, 400)
        self.fc2 = nn.Linear(400, 800)
        self.fc3 = nn.Linear(800, 256)
        self.fc4 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.tanh(x)
        return x


class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 2)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x


def load_model_from_file(model_path, ModelClass):
    model = ModelClass()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def train(model, traindata, optimizer=None, lr=0.0001, weight_decay=0,
          batch_size=32, total_epoch=25, num_workers=0):
    criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_count = 0
        for i, data in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # get the inputs; data is a list of [inputs, labels]
            # inputs is a list of data with len(inputs) transform variances applied on the batch data
            inputs, labels = data
            # reshape labels to be in batch format
            labels = labels.unsqueeze(-1).type(torch.FloatTensor)

            # forward + backward + optimize
            loss = torch.tensor(0).type(torch.FloatTensor)
            for j in range(len(inputs)):
                outputs = model(inputs[j])
                outputs = outputs.type(torch.FloatTensor)
                loss += criterion(outputs, labels)
            loss = loss/len(inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_count += 1
        print(f'[{epoch + 1}] loss: {running_loss / batch_count:.3f}')
        print(batch_count)
    optimizer.zero_grad()
    return optimizer
