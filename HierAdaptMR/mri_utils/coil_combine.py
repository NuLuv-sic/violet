import torch

def rss(data, dim=0):
    """
    Root Sum of Squares (RSS) 合并。
    输入 data 通常为 [Coil, H, W] 或 [Coil, H, W, 2]
    """
    return torch.sqrt((data**2).sum(dim))

def rss_complex(data, dim=0):
    """
    针对复数数据的 RSS 合并。
    输入 data 为 [Coil, H, W, 2]
    """
    # 先计算复数的模：sqrt(real^2 + imag^2)
    model = torch.sqrt(data.pow(2).sum(-1))
    # 再对线圈维度做 RSS
    return torch.sqrt(model.pow(2).sum(dim))
