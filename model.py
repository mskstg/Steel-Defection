import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 50 * 8, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        x = self.pool(F.relu(self.conv3(x.float())))
        x = self.pool(F.relu(self.conv4(x.float())))
        x = self.pool(F.relu(self.conv5(x.float())))
        x = x.view(-1, 128 * 50 * 8)
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x.float()))
        x = torch.sigmoid(self.fc3(x.float()))
        return x