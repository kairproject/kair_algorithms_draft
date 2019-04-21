import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):

    def __init__(self, in_channel, action_dim, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, 3, (3, 3, 3), 2, 1)
        self.conv2 = nn.Conv3d(3, 128, (3, 3, 3), 2, 1)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), 2, 1)
        self.avg_pool4 = nn.AvgPool3d(5)
        self.fc5 = nn.Linear(256, 512)
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        self.fc5.bias.data.uniform_(-3e-3, 3e-3)
        self.fc6 = nn.Linear(512, action_dim)
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)
        self.fc6.bias.data.uniform_(-3e-3, 3e-3)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avg_pool4(x)
        x = F.relu(self.fc5(x.view(x.size(0), -1)))
        x = self.fc6(x)
        x = self.max_action * torch.tanh(x)
        return x.squeeze()


class Critic(nn.Module):

    def __init__(self, in_channel, action_dim):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv3d(in_channel, 3, (3, 3, 3), 2, 1)
        self.conv2 = nn.Conv3d(3, 128, (3, 3, 3), 2, 1)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), 2, 1)
        self.avg_pool4 = nn.AvgPool3d(5)

        self.fc5 = nn.Linear(256 + action_dim, 512)
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        self.fc5.bias.data.uniform_(-3e-3, 3e-3)
        self.fc6 = nn.Linear(512, 1)
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)
        self.fc6.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, u):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avg_pool4(x)
        xu = torch.cat([x.view(x.size(0), -1), u], 1)

        x = F.relu(self.fc5(xu))
        x = self.fc6(x)
        return x
