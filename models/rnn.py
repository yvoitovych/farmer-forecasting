import numpy as np
import pandas as pd
# import cv2
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = torch.nn.GRU(40, 40, 4, batch_first=True)
        self.sigm =  nn.Sigmoid()

    def forward(self, x):
        h, x = self.recurrent(x)

        return x[-1]


# net = Net()
