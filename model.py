import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


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


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2, 256, 3, padding="same")
#         self.conv2 = nn.Conv2d(256, 256, 3)
#         self.conv3 = nn.Conv2d(256, 16, 1)
#         self.fc1 = nn.Linear(400, 800)
#         self.fc2 = nn.Linear(800, 256)
#         self.fc3 = nn.Linear(256, 1)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         # print(np.abs(x.detach().numpy().std(axis=0)).mean())
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = torch.flatten(x, 1)     # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.tanh(x)
#         # print(np.abs(x.detach().numpy().std(axis=0)).mean())
#         # print("-------------")
#         return x


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
        # print(np.abs(x.detach().numpy().std(axis=0)).mean())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        # print(np.abs(x.detach().numpy().std(axis=0)).mean())
        # print("-------------")
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

    def test(self, x):
        return self.fc2(x)


class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 2)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print(np.abs(x.detach().numpy().std(axis=0)).mean())
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.tanh(x)
        # print(np.abs(x.detach().numpy().std(axis=0)).mean())
        # print("-------------")
        return x


def load_model_from_file(model_path, ModelClass):
    model = ModelClass()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def train(model, traindata, batch_size=32, lr=0.0001, weight_decay=0, num_workers=0, total_epoch=25):
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_count = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(np.abs(labels.detach().numpy().std(axis=0)).mean())
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # reshape labels to be in batch format
            labels = torch.reshape(labels, outputs.shape).type(torch.FloatTensor)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_count += 1
        print(f'[{epoch + 1}] loss: {running_loss / batch_count:.3f}')
