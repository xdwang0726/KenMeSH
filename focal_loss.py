import torch
import torch.nn as nn
import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha=.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         # self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         # BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#         # print('bce', BCE_loss.shape)
#         # targets = targets.type(torch.long)
#         # at = self.alpha.gather(0, targets.data.view(-1))
#         # print('at', )
#         # pt = torch.exp(-BCE_loss)
#         # F_loss = at*(1-pt)**self.gamma * BCE_loss
#         # print('bce', BCE_loss.shape)
#
#         # zeros = torch.zeros_like(inputs)
#         # pos_p_sub = torch.where(targets > zeros, targets - inputs, zeros)
#         # neg_p_sub = torch.where(targets > zeros, zeros, inputs)
#         # per_entry_cross_ent = -self.alpha * (pos_p_sub ** self.gamma) * torch.log(torch.clamp(inputs, 1e-8, 1.0)) - (
#         #             1 - self.alpha) * (neg_p_sub ** self.gamma) * torch.log(torch.clamp(1.0 - inputs, 1e-8, 1.0))
#
#         return per_entry_cross_ent.mean()


class FocalLoss(nn.Module):
    """FocalLoss.
    .. seealso::
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        Proceedings of the IEEE international conference on computer vision. 2017.
    Args:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.
    Attributes:
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
        eps (float): Epsilon to avoid division by zero.
    """
    def __init__(self, gamma=2, alpha=0.25, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, input, target):
        input = input.clamp(self.eps, 1. - self.eps)

        cross_entropy = - (target * torch.log(input) + (1 - target) * torch.log(1 - input))  # eq1
        logpt = - cross_entropy
        pt = torch.exp(logpt)  # eq2

        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        balanced_cross_entropy = - at * logpt  # eq3

        focal_loss = balanced_cross_entropy * ((1 - pt) ** self.gamma)  # eq5

        return focal_loss.sum()



