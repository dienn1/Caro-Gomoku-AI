import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(np.abs(x.detach().numpy().std(axis=0)).mean())
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def train(model, traindata, batch_size=32, lr=0.0001, num_workers=0, total_epoch=25):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for epoch in range(total_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_count = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = torch.reshape(outputs, labels.shape).type(torch.FloatTensor)
            # print(outputs[:10], labels[:10])
            loss = criterion(outputs, labels.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_count += 1
        print(f'[{epoch + 1}] loss: {running_loss / batch_count:.3f}')
    #print(list(model.parameters())[-2])
