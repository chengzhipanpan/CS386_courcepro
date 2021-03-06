from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, weight=[1.0] * 8):
        super(Loss, self).__init__()
        self.weight = weight

    def forward(self, x_list, label):
        loss = self.weight[0] * F.binary_cross_entropy(x_list[0], label)
        i = 1
        for x in x_list[1:-1]:
            loss += self.weight[i + 1] * F.binary_cross_entropy(x, label)
        loss+=2*F.binary_cross_entropy(x_list[-1],label)
        return loss