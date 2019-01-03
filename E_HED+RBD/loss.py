from torch import nn
import torch.nn.functional as F


# Directly use the cross entropy loss
class Loss(nn.Module):
    def __init__(self, weight=[1.0] * 8):
        super(Loss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        loss = F.binary_cross_entropy(x_list, label)
        return loss