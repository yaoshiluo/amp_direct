# soft_dtw_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDTWBatch(nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x, y):
        # x: [B, T, D], y: [B, T, D]
        B, T, D = x.size()
        D_xy = torch.cdist(x, y, p=2) ** 2  # [B, T, T]
        R = torch.zeros((B, T + 1, T + 1), device=x.device) + float("inf")
        R[:, 0, 0] = 0.0

        for i in range(1, T + 1):
            for j in range(1, T + 1):
                r0 = R[:, i - 1, j - 1]
                r1 = R[:, i - 1, j]
                r2 = R[:, i, j - 1]
                rmin = torch.stack([r0, r1, r2], dim=-1)
                R[:, i, j] = D_xy[:, i - 1, j - 1] + (-self.gamma * torch.logsumexp(-rmin / self.gamma, dim=-1))

        loss = R[:, T, T]
        if self.normalize:
            loss = loss / T
        return loss
