import torch.nn.functional as F

def encoder_loss(rep, mask_rep, pred, mask_pred, label, alpha):
    return F.cross_entropy(pred, label) + F.cross_entropy(mask_pred, label) + alpha * F.mse_loss(rep, mask_rep, reduction='sum')