# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
class CIFARNet(nn.Module):
    def __init__(self, num_class, num_channel=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3, padding=1),  # 32 X 32 X 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 X 32 X 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 X 32 X 32

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # output: 128 X 16 X 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 X 8 X 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # output: 256 X 8 X 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 X 4 X 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_class),
            nn.Softmax()
        )

    def forward(self, x):
        return self.network(x)