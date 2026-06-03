import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.tools import LambdaSheduler



class CMKD(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lamb = LambdaSheduler(max_iter=args.max_iter)
        self.args = args


    """
        计算两个概率分布之间的相似度，并转化为 0~1 之间的置信度系数。
        使用 KL 散度 (Kullback-Leibler Divergence) 计算视觉分类器预测 和 文本/预训练预测 之间的距离
        将距离映射为系数：coe 如果两者预测高度一致，KL散度接近0，coe 接近1；如果分歧很大，coe 趋近于0。

    """
    def calibrated_coefficient(self, pred, pred_pretrained):
        distance = F.kl_div(pred.log(), pred_pretrained, reduction='none').sum(-1)
        coe = torch.exp(-distance).detach()
        return coe


    def gini_impurity(self,pred,coe=1.0):
        sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
        return torch.sum(coe * (1 - torch.sum(pred ** 2 / sum_dim, dim=-1)))


    def regularization_term(self, target_txt_pred, source_txt_logit, source_label,lamb):
        return self.args.lambda2*F.cross_entropy(source_txt_logit, source_label) + \
            self.args.lambda3*lamb*self.gini_impurity(target_txt_pred)


    def forward(self, target_cls_logit, target_txt_logit, source_txt_logit, source_label):
        target_cls_pred = F.softmax(target_cls_logit, dim=1)
        target_txt_pred = F.softmax(target_txt_logit,dim=-1)
        coe = self.calibrated_coefficient(target_cls_pred, target_txt_pred)
        target_pred_mix = 0.5*(target_cls_pred+target_txt_pred.detach())
        lamb = self.lamb.lamb()
        task_loss = self.args.lambda1 * lamb * self.gini_impurity(target_cls_pred,coe)
        distill_loss = self.args.lambda1 * lamb *self.gini_impurity(target_pred_mix,1-coe)
        reg_loss = self.regularization_term(target_txt_pred, source_txt_logit, source_label,lamb)
        self.lamb.step()
        return task_loss + distill_loss + reg_loss

