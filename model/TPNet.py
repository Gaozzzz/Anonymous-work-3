import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from .module.memory import Memory_Unit
from .module.transformer import Transformer


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=out_size, hidden_size=out_size, num_layers=1,
                            bidirectional=True, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x


class network(Module):
    def __init__(self, input_size, flag, nums_block):
        super().__init__()
        self.flag = flag
        self.embedding = Temporal(input_size, 512)
        self.selfatt = Transformer(1024, 2, 4, 128, 512, dropout=0.5)
        self.Memory = Memory_Unit(nums=nums_block, dim=1024)
        self.cls_head = nn.Sequential(nn.Linear(2048, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        if self.flag == "Train":
            M_x = self.Memory(x)
            x = torch.cat((x, M_x), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}
        else:
            M_x = self.Memory(x)
            x = torch.cat([x, M_x], dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}
