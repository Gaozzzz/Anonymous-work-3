import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class Memory_Unit(Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim
        self.nums = nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        self.sig = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data):
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim ** 0.5))
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block)
        return augment
