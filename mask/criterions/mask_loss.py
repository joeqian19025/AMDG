import torch
import torch.nn.functional as F
import numpy as np


def mask_regularizer(mask):
    return 1 / torch.sin(np.pi * torch.mean(mask))


def classifier_loss(label, masked_output, inverse_masked_output):
    return F.cross_entropy(masked_output, label) + F.cross_entropy(
        inverse_masked_output, label
    )


def mask_loss(label, masked_output, inverse_masked_output, mask):
    return (
        F.cross_entropy(masked_output, label)
        - F.cross_entropy(inverse_masked_output, label)
        + mask_regularizer(mask)
    )
