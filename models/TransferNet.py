import torch
import torch.nn as nn
from models.backbone import CLIP
from models import cmkd, cvcd
import logging
import torch.nn.functional as F
import copy



"""
    处理分类器的具体层内逻辑：
    对于线性层，使用正态分布进行权重初始化，标准差为0.001。
    对于批归一化层，禁用偏置项的梯度更新，并将权重初始化为1.0，偏置初始化为0.0。
"""
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# 对clip的BN层进行冻结，保持其在训练过程中不更新，以稳定训练过程并防止过拟合。
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
       m.eval()



class TransferNet(nn.Module):
    def __init__(self, args, train=True):
        super(TransferNet, self).__init__()
        self.args = args
        self.num_class = args.num_class
        self.base_network = CLIP(args).to(args.device)
        # self.teacher_model = copy.deepcopy(self.base_network).to(args.device)
        # self.teacher_model.eval()
        # 我不知道为什么源代码定义了教师模型但不用，纯纯耗老子显存

        # define the task head
        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.base_network.output_num),
            nn.LayerNorm(self.base_network.output_num, eps=1e-6),
            nn.Linear(self.base_network.output_num, self.num_class,bias=False))
        self.classifier_layer.apply(weights_init_classifier)

        if train:
            # define the loss functions
            self.cmkd = cmkd.CMKD(args)
            self.cvcd = cvcd.CVCD(args)
            self.clf_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)



    """
        获取模型参数，返回一个包含模型参数的列表，以便优化器使用。
    """
    def get_parameters(self, initial_lr=1.0):
        params=[
            {'params': self.base_network.model.visual.parameters(), 'lr': initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': self.args.multiple_lr_classifier * initial_lr}
        ]
        return params



    """
        对多视角特征进行池化操作，
        这里可以选择不同的方式，暂时选择maxpooling来做
    """
    def target_feature_pooling(self, features_multiviews):
        pooled_features, _ = torch.max(features_multiviews, dim=1)    # 对多视角特征进行最大池化，得到每个样本的整体特征表示，这里得到的是f(x)
        return pooled_features



    """
        前向传播
        计算并返回分类头的损失和跨域一致性蒸馏损失
    """
    def forward(self, source_imgs, source_labels, target_imgs, target_strong_imgs=None):
        self.base_network.apply(fix_bn)     # 冻结clip的BN层

        # 处理源域图像的多视角情况
        if len(source_imgs.shape) == 5:       # [B, V, C, H, W]
            B, V, C, H, W = source_imgs.shape
            source_imgs_flat = source_imgs.view(B*V, C, H, W)
            source_features_flat = self.base_network.forward_features(source_imgs_flat)
            source_features_multiviews = source_features_flat.view(B, V, -1)
            source_features = self.target_feature_pooling(source_features_multiviews)    # 对多视角特征进行池化操作，得到每个样本的整体特征表示
        else:
            source_features = self.base_network.forward_features(source_imgs)  # 提取源域图像的特征
        # source_features = self.base_network.forward_features(source_imgs)               # 提取源域图像的特征，这里得到的是f(x)

        source_cls_logits = self.classifier_layer(source_features)                      # 通过分类头得到源域图像的预测结果logits，这里得到的是ph'
        clf_loss = self.clf_loss(source_cls_logits, source_labels)                      # 计算分类损失，这里仅仅是拿源域来训练clip的视觉编码器，和分类头。没有文本编码器的参与，也不涉及目标域

        source_txt_logits = self.base_network.forward_head(source_features)             # 计算源域图像特征与文本提示之间的余弦相似度

        if len(target_imgs.shape) == 5:       # [B, V, C, H, W]，说明这里的目标域是多视角情况，我们需要先展开来进行特征提取和和pooling
            B, V, C, H, W = target_imgs.shape
            target_imgs_flat = target_imgs.view(B*V, C, H, W)     # 展开成[B*V, C, H, W]
            target_features_flat = self.base_network.forward_features(target_imgs_flat)         # 提取目标域图像的特征，这里得到的是f(x)，它的形状是[B*V, feature_dim]
            target_features_multiviews = target_features_flat.view(B, V, -1)                    # 重新调整形状为[B, V, feature_dim]
            target_features_global = self.target_feature_pooling(target_features_multiviews)    # 对多视角特征进行池化操作，得到每个样本的整体特征表示，它的形状是[B, feature_dim]

            # 上诉过程获得每张view的特征和pooling后的全局特征，接下来我们计算各式logit和损失
            target_cls_logits_global = self.classifier_layer(target_features_global)            # 通过分类头得到目标域图像的预测结果logits
            target_txt_logits_global = self.base_network.forward_head(target_features_global)   # 计算目标域图像特征与文本提示之间的余弦相似度
            transfer_loss = self.cmkd(target_cls_logits_global, target_txt_logits_global, source_txt_logits, source_labels)     # 计算跨模态知识蒸馏损失

            if self.args.cvcd:      # 跨视角一致性蒸馏
                target_cls_logits_multiviews = self.classifier_layer(target_features_flat)    # 通过分类头得到多视角特征的预测结果logits，形状为[B*V, num_class]
                cvcd_loss = self.cvcd(target_cls_logits_global, target_cls_logits_multiviews)
                transfer_loss += cvcd_loss
        
        else:       # 这里的目标域是单视角情况，直接提取特征和计算损失  
            target_features_global = self.base_network.forward_features(target_imgs)               # 提取目标域图像的特征，这里得到的是f(x)，它的形状是[B, feature_dim]
            target_cls_logits_global = self.classifier_layer(target_features_global)                      # 通过分类头得到目标域图像的预测结果logits，这里得到的是ph'
            target_txt_logits_global = self.base_network.forward_head(target_features_global)             # 计算目标域图像特征与文本提示之间的余弦相似度
            transfer_loss = self.cmkd(target_cls_logits_global, target_txt_logits_global, source_txt_logits, source_labels)     # 计算跨模态知识蒸馏损失


        # 如果用了fixmatch，他会用强增强的目标域图像来计算损失，这样可以利用无标签数据来进一步提升模型的性能
        if self.args.fixmatch:
            assert target_strong_imgs is not None, "FixMatch requires strong augmented target images"
            # 获取弱增强生成的伪标签 (Shape: [B]) 和置信度权重 (Shape: [B])
            target_pred_cls = F.softmax(target_cls_logits_global, dim=-1)
            max_prob, pred_u = torch.max(target_pred_cls, dim=-1)
            # 伪标签和权重不需要梯度
            pred_u = pred_u.detach()
            fixmatch_mask = max_prob.ge(self.args.fixmatch_threshold).float().detach()

            if len(target_strong_imgs.shape) == 5:         # [B, V, C, H, W]，说明这里的目标域是多视角情况
                B, V, C, H, W = target_strong_imgs.shape
                target_strong_imgs_flat = target_strong_imgs.view(B*V, C, H, W)     # 展开成[B*V, C, H, W]
                target_strong_features_flat = self.base_network.forward_features(target_strong_imgs_flat)         # 提取强增强目标域图像的特征，这里得到的是f(x)，它的形状是[B*V, feature_dim]
                target_strong_features_multiviews = target_strong_features_flat.view(B, V, -1)                    # 重新调整形状为[B, V, feature_dim]
                target_strong_features_global = self.target_feature_pooling(target_strong_features_multiviews)    # 对多视角特征进行池化操作得到样本的整体特征表示，它的形状是[B, feature_dim]
                target_strong_cls_logits_global = self.classifier_layer(target_strong_features_global)            # 通过分类头得到强增强目标域图像的预测结果logits
            else:    # 这里的目标域是单视角情况
                target_strong_features_global = self.base_network.forward_features(target_strong_imgs)               # 提取强增强目标域图像的特征，这里得到的是f(x)，它的形状是[B, feature_dim]
                target_strong_cls_logits_global = self.classifier_layer(target_strong_features_global)                      # 通过分类头得到强增强目标域图像的预测结果logits，这里得到的是ph'   

            fixmatch_loss = self.args.fixmatch_factor * (
                F.cross_entropy(target_strong_cls_logits_global, pred_u, reduction='none') * fixmatch_mask
            ).mean()

            # 接下来计算clip的文本logit的fixmatch 损失
            target_pred_txt = F.softmax(target_txt_logits_global, dim=1)
            max_prob, pred_u = torch.max(target_pred_txt, dim=-1)
            pred_u = pred_u.detach()
            fixmatch_mask_txt = max_prob.ge(self.args.fixmatch_threshold).float().detach()
            target_strong_txt_logits_global = self.base_network.forward_head(target_strong_features_global)
            fixmatch_loss += self.args.fixmatch_factor * (
                F.cross_entropy(target_strong_txt_logits_global, pred_u, reduction='none') * fixmatch_mask_txt
            ).mean()

            transfer_loss += fixmatch_loss
        
        return clf_loss, transfer_loss



    """
        推理阶段直接进行分类预测
    """
    def predict(self, imgs):
        if len(imgs.shape) == 5:       # [B, V, C, H, W]，说明这里的输入是多视角情况
            B, V, C, H, W = imgs.shape
            x_flat = imgs.view(B*V, C, H, W)     # 展开成[B*V, C, H, W]
            features_flat = self.base_network.forward_features(x_flat)         # 提取特征，这里得到的是f(x)，它的形状是[B*V, feature_dim]
            features_multiviews = features_flat.view(B, V, -1)                    # 重新调整形状为[B, V, feature_dim]
            features_global = self.target_feature_pooling(features_multiviews)    # 对多视角特征进行池化操作，得到每个样本的整体特征表示，它的形状是[B, feature_dim]
            cls_logits_global = self.classifier_layer(features_global)            # 通过分类头得到预测结果logits
            return cls_logits_global
        
        else:       # 这里的输入是单视角情况，直接提取特征和计算logits  
            features_global = self.base_network.forward_features(imgs)               # 提取特征，这里得到的是f(x)，它的形状是[B, feature_dim]
            cls_logits_global = self.classifier_layer(features_global)                      # 通过分类头得到预测结果logits，这里得到的是ph'
            return cls_logits_global



