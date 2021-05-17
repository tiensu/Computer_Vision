import numpy as np
import torch
import torch.nn as tn
import torch.nn.functional as tnf


class ContrastiveLoss(tn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = tnf.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
