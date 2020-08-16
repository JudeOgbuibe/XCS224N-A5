#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, emb_size):
        super(Highway, self).__init__()
        self.emb_size = emb_size

        self.linear_proj = nn.Linear(self.emb_size, self.emb_size)
        self.linear_gate = nn.Linear(self.emb_size, self.emb_size)


    def forward(self, X):
        X_proj = torch.relu(self.linear_gate(X))
        X_gate = torch.sigmoid(self.linear_gate(X))
        X_highway = torch.mul(X_gate, X_proj) + torch.mul((1 - X_gate), X)

        return X_highway
### END YOUR CODE 

