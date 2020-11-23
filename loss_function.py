import torch
import torch.nn as nn

class gradient_norm_loss(nn.Module):

    def __init__(self):
        super(gradient_norm_loss, self).__init__()

    def forward(self, gradient_tensor):
        return 1 / (torch.norm(gradient_tensor))