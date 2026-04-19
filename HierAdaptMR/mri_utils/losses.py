import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    修正后的 SSIM Loss，增加了 padding 和 groups 支持。
    """
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.padding = win_size // 2  # 关键：确保卷积后尺寸不变
        
        # 将权重设为可注册缓存
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / (win_size ** 2))

    def forward(self, X, Y, data_range=None):
        if data_range is None:
            data_range = Y.max() - Y.min()

        # 确保输入是 4D [B, C, H, W]
        if len(X.shape) == 3:
            X, Y = X.unsqueeze(1), Y.unsqueeze(1)
            
        b, c, h, w = X.shape
        # 必须指定 groups=c，否则会报错或计算错误
        # 使用 padding=self.padding 保持维度
        conv_args = {'groups': c, 'padding': self.padding}

        mu1 = F.conv2d(X, self.w.expand(c, -1, -1, -1), **conv_args)
        mu2 = F.conv2d(Y, self.w.expand(c, -1, -1, -1), **conv_args)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(X * X, self.w.expand(c, -1, -1, -1), **conv_args) - mu1_sq
        sigma2_sq = F.conv2d(Y * Y, self.w.expand(c, -1, -1, -1), **conv_args) - mu2_sq
        sigma12 = F.conv2d(X * Y, self.w.expand(c, -1, -1, -1), **conv_args) - mu1_mu2

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()
