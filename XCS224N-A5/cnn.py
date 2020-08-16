#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, kernel_size, filter_size, char_emb_size):
        super(CNN, self).__init__()
        self.char_emb_size = char_emb_size
        self.kernel_size = kernel_size
        self.filter_size = filter_size

        self.conv1d = nn.Conv1d(in_channels=self.char_emb_size, out_channels=self.filter_size,
                            kernel_size=self.kernel_size)
        self.max_pool = nn.MaxPool1d(17)

    def forward(self, X):
        X_conv = self.conv1d(X)
        #print(X_conv.size())
        X_conv = self.max_pool(torch.relu(X_conv))
        #print(X_conv.size())
        return X_conv.squeeze()
 
### END YOUR CODE

