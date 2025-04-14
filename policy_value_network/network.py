"""
AlphaZero-style policy-value network
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from enum import Enum
from engine.game import Action


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class BombermanCNN(nn.Module):
    def __init__(self, in_channels=17, num_actions=6, board_h=9, board_w=11,
                 num_blocks=8, channels=128):
        super(BombermanCNN, self).__init__()
        self.board_h = board_h
        self.board_w = board_w

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_h * board_w, num_actions)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_h * board_w, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.res_blocks(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value