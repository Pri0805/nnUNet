import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

#### EDITS MADE FOR TVERKSEY LOSS ####

class TverskyLoss(RobustCrossEntropyLoss):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, **kwargs):
        super(TverskyLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, net_output, target):
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        
        target_one_hot = torch.zeros_like(net_output)
        target_one_hot.scatter_(1, target.long(), 1)

        net_output = net_output.contiguous().view(-1)
        target_one_hot = target_one_hot.contiguous().view(-1)

        true_pos = torch.sum(target_one_hot * net_output)
        false_neg = torch.sum(target_one_hot * (1 - net_output))
        false_pos = torch.sum((1 - target_one_hot) * net_output)

        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky_index
