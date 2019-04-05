import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 10)
        self.loss = 0.0
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_loss(self):
        return self.loss

    def set_loss(self, loss):
        self.loss = loss

    def set_parameter(self, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b):
        self.conv1.weight = torch.nn.Parameter(torch.Tensor(conv1_w))
        self.conv1.bias = torch.nn.Parameter(torch.Tensor(conv1_b))
        self.conv2.weight = torch.nn.Parameter(torch.Tensor(conv2_w))
        self.conv2.bias = torch.nn.Parameter(torch.Tensor(conv2_b))
        self.fc1.weight = torch.nn.Parameter(torch.Tensor(fc1_w))
        self.fc1.bias = torch.nn.Parameter(torch.Tensor(fc1_b))

# net = ConvNet()
