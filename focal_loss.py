import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        # self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # print('bce', BCE_loss.shape)
        # targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        # print('at', )
        # pt = torch.exp(-BCE_loss)
        # F_loss = at*(1-pt)**self.gamma * BCE_loss
        # print('bce', BCE_loss.shape)
        zeros = torch.zeros_like(inputs)
        pos_p_sub = torch.where(targets > zeros, targets - inputs, zeros)
        neg_p_sub = torch.where(targets > zeros, zeros, inputs)
        per_entry_cross_ent = -self.alpha * (pos_p_sub ** self.gamma) * torch.log(torch.clamp(inputs, 1e-8, 1.0)) - (
                    1 - self.alpha) * (neg_p_sub ** self.gamma) * torch.log(torch.clamp(1.0 - inputs, 1e-8, 1.0))
        return per_entry_cross_ent.sum()



