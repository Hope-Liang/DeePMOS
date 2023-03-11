import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightExponentialMovingAverage:

    def __init__(self, net, momentum_net, alpha=0.99):
        self.alpha = alpha
        self.params = list(net.state_dict().values())
        self.momentum_params = list(momentum_net.state_dict().values())

    def step(self):
        for param, momentum_param in zip(self.params, self.momentum_params):
            momentum_param.copy_(momentum_param.data*self.alpha + param.data*(1.0 - self.alpha))
            
    def set_alpha(self, alpha):
        self.alpha = alpha