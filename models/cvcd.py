import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.tools import LambdaSheduler


class CVCD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lamb = LambdaSheduler(max_iter=args.max_iter)
        self.args = args

    def forward(self, target_logit, target_logit_views, label_set=None):
        pass