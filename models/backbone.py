import torch.nn as nn
import clip



class CLIP(nn.Module):
    def __init__(self,args):
        super(CLIP, self).__init__()
        if args.model_name == "RN50":
            model, preprocess = clip.load("RN50", device=args.device, download_root="/data1/dengxuxiang/.cache/clip")
            self.output_num = 1024
        elif args.model_name == "RN101":
            model, preprocess = clip.load("RN101", device=args.device, download_root="/data1/dengxuxiang/.cache/clip")
            self.output_num = 512
        elif args.model_name == "VIT-B":
            model, preprocess = clip.load("ViT-B/16", device=args.device, download_root="/data1/dengxuxiang/.cache/clip")
            self.output_num = 512
        elif args.model_name == "VIT-L":
            model, preprocess = clip.load("ViT-L/14", device=args.device, download_root="/data1/dengxuxiang/.cache/clip")
            self.output_num = 768
        model = model.float()       # clip直接load的模型是fp16的，这里转成fp32
        if args.datasets=="MI3DOR":
            class_list = ['a image of a airplane', 'a image of a bed', 'a image of a bicycle', 'a image of a bookshelf', 'a image of a camera', 'a image of a car', 'a image of a chair', 'a image of a flower pot', 'a image of a guitar', 'a image of a keyboard', 'a image of a knife', 'a image of a monitor', 'a image of a motorcycle', 'a image of a pistol', 'a image of a plant', 'a image of a radio', 'a image of a rifle', 'a image of a stairs', 'a image of a tent', 'a image of a vase', 'a image of a wardrobe']
        elif args.datasets=="MI3DOR-2":
            class_list = ['a image of a airplane', 'a image of a bathtub', 'a image of a bed', 'a image of a bench', 'a image of a bookshelf', 'a image of a bottle', 'a image of a bowl', 'a image of a car', 'a image of a chair', 'a image of a cone', 'a image of a cup', 'a image of a curtain', 'a image of a desk', 'a image of a door', 'a image of a dresser', 'a image of a flower pot', 'a image of a glass box', 'a image of a guitar', 'a image of a keyboard', 'a image of a lamp', 'a image of a laptop', 'a image of a mantel', 'a image of a monitor', 'a image of a night stand', 'a image of a person', 'a image of a piano', 'a image of a plant', 'a image of a radio', 'a image of a range hood', 'a image of a sink', 'a image of a sofa', 'a image of a stairs', 'a image of a stool', 'a image of a table', 'a image of a tent', 'a image of a toilet', 'a image of a tv stand', 'a image of a vase', 'a image of a wardrobe', 'a image of a xbox']

        self.model = model
        self.args = args
        self.text = clip.tokenize(class_list).to(args.device)
        text_features = self.encode_text().detach().to(args.device)          # 这里得到的是文本特征 g(t) 
        self.text_features = text_features / text_features.norm(dim=1, keepdim=True)    # 文本特征归一化


    # 用clip的视觉编码器提取图像特征
    def forward_features(self, x):
        feature = self.model.encode_image(x)
        return feature


    # 用clip的文本编码器提取文本特征
    def encode_text(self):
        text_features = self.model.encode_text(self.text)
        return text_features


    """
        计算图像特征与文本提示（Prompt）之间的逻辑分数。
        一般情况下返回的是图像特征与文本提示之间的相似度分数（logits_per_image），
        但如果参数return_text_logit为True，则同时返回文本特征与图像提示之间的相似度分数（logits_per_text）。
        这个函数得到的结果是图像特征和文本特征的余弦相似度结果，这个将会之后用于公式2的后续计算。
    """
    def forward_head(self,image_features, return_text_logit=False):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)      # 图像特征归一化
        text_features = self.text_features

        logit_scale = self.model.logit_scale.exp()      # 可学习的缩放因子 logit_scale
        logits_per_image = logit_scale * image_features @ text_features.t()     # 由于图像和文本特征都进行了归一化，那么这里得到的是余弦相似度，这里的结果是公式2中的余弦相似度结果
        logits_per_text = logits_per_image.t()
        if return_text_logit:
            return logits_per_image, logits_per_text
        else:
            return logits_per_image


    # 综合forward_features和forward_head，直接得到图像输入对应的分类分数（logits_per_image）
    def forward(self, x):
        image_features = self.forward_features(x)
        logits_per_image = self.forward_head(image_features)

        return logits_per_image

