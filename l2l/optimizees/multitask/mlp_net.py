import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.lin1 = nn.Linear(784, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 10)
        self.loss = 0.0

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)

    def set_parameter(self, lin1_w, lin1_b, lin2_w, lin2_b, lin3_w, lin3_b):
        self.lin1.weight = torch.nn.Parameter(torch.Tensor(lin1_w))
        self.lin1.bias = torch.nn.Parameter(torch.Tensor(lin1_b))
        self.lin2.weight = torch.nn.Parameter(torch.Tensor(lin2_w))
        self.lin2.bias = torch.nn.Parameter(torch.Tensor(lin2_b))
        self.lin3.weight = torch.nn.Parameter(torch.Tensor(lin3_w))
        self.lin3.bias = torch.nn.Parameter(torch.Tensor(lin3_b))

    def get_loss(self):
        return self.loss

    def set_loss(self, loss):
        self.loss = loss
