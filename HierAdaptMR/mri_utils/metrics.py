import torch
import numpy as np

def nmse(gt, pred):
    return torch.norm(gt - pred) ** 2 / torch.norm(gt) ** 2

def psnr(gt, pred):
    mse = torch.mean((gt - pred) ** 2)
    if mse == 0: return 100
    max_val = gt.max()
    return 20 * torch.log10(max_val / torch.sqrt(mse))
