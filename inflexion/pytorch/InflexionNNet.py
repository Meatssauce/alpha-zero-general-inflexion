import sys

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn

sys.path.append('..')


class SelfPlayDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, i):
        example = self.examples[i]
        return (
            torch.tensor(example[0]).float(),
            torch.tensor(example[1]),
            torch.tensor(example[2]).float(),
        )

    def __len__(self):
        return len(self.examples)


class SumModule(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.branches = nn.ModuleList(modules)

    def forward(self, x):
        return sum(module(x) for module in self.branches)


class InflexionNNet(nn.Module):
    def __init__(self, game):
        # game params
        self.depth, self.board_x, self.board_y = game.to_planes().shape
        self.pi_shape = game.policy_shape
        self.max_actions = game.max_actions
        assert self.max_actions <= 343

        super(InflexionNNet, self).__init__()
        self.residual_tower = nn.Sequential(
            nn.Conv2d(self.depth, 16, 3, 1, 1),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(16, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.Mish(),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                ),
                nn.Conv2d(16, 32, 1, 1, 0),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(32, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.Mish(),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                ),
                nn.Conv2d(32, 64, 1, 1, 0),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.Mish(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                ),
                nn.Conv2d(64, 128, 1, 1, 0),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.Mish(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                ),
                nn.Sequential(),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.Mish(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                ),
                nn.Sequential(),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.Mish(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                ),
                nn.Sequential(),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.Mish(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                ),
                nn.Sequential(),
            ),
            nn.Mish(),
            SumModule(
                nn.Sequential(
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.Mish(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                ),
                nn.Sequential(),
            ),
            nn.Mish(),
        )

        self.pi_tower = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.Mish(),
            nn.Linear(512, self.max_actions),
            nn.LogSoftmax(dim=1),
        )

        self.v_tower = nn.Sequential(
            nn.Conv2d(128, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.eval()

    def forward(self, s):
        x = self.residual_tower(s)
        pi = self.pi_tower(x)
        v = self.v_tower(x)
        return pi, v
