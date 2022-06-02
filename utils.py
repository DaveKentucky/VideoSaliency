import torch
from torch import nn
import sys


def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp / torch.sum(inp)
        trg = trg / torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg * torch.log(eps + torch.div(trg, (inp + eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)
