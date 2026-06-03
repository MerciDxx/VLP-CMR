from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from utils.FixMatch import TransformFixMatch



"""
    单视角数据集，每个样本返回一张图像和对应标签
    适用于源域训练和目标域测试
"""
class SingleViewDataset(Dataset):
    def __init__(self, index_txt, transform=None):
        with open(index_txt, 'r') as f:
            image_list = f.readlines()

        self.classes = set()
        self.imgs = []
        for val in image_list:
            path, label = val.split()[0], int(val.split()[1])
            self.imgs.append((path, label))
            if label not in self.classes:
                self.classes.add(label)

        if len(self.imgs) == 0:
            raise(RuntimeError("no image"))
        self.transform = transform


    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.imgs)



"""
    多视角数据集，每个样本返回同一图像的多张视图和对应标签
    支持选择其中某几个视角
"""
class MultiViewDataset(Dataset):
    def __init__(self, index_txt, transform=None, use_fixmatch=False, views_index=None):
        with open(index_txt, 'r') as f:
            image_list = f.readlines()

        self.classes = set()
        self.imgs = []
        for val in image_list:
            path, label = val.split()[0], int(val.split()[1])
            self.imgs.append((path, label))
            if label not in self.classes:
                self.classes.add(label)

        if len(self.imgs) == 0:
            raise(RuntimeError("no image"))
        self.use_fixmatch = use_fixmatch
        self.transform = transform
        self.views_index = views_index      # 支持选择其中某几个视角，默认使用全部12个视角
        if self.views_index is None:
            self.views_index = list(range(12))


    def __getitem__(self, index):
        start_idx = index * 12
        if self.use_fixmatch:
            assert self.transform is not None, "FixMatch Transform must be provided when use_fixmatch is True"
            weak_imgs = []
            strong_imgs = []
            label = None

            for i in self.views_index:
                path, lab = self.imgs[start_idx + i]
                img = Image.open(path).convert("RGB")
                # 此时 transform 返回 (weak_img, strong_img)
                weak_img, strong_img = self.transform(img)
                weak_imgs.append(weak_img)
                strong_imgs.append(strong_img)
                if label is None:
                    label = lab

            weak_imgs = torch.stack(weak_imgs)
            strong_imgs = torch.stack(strong_imgs)
            return (weak_imgs, strong_imgs), label
        
        else:
            imgs = []
            label = None

            for i in self.views_index:
                path, lab = self.imgs[start_idx + i]
                img = Image.open(path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append(img)

                if label is None:
                    label = lab

            imgs = torch.stack(imgs)
            return imgs, label


    def __len__(self):
        return len(self.imgs) // 12



"""
    数据加载器函数，支持普通数据加载器和迭代式数据加载器
    迭代式数据加载器在每个epoch结束后会重新打乱数据，适用于源域训练
"""
def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0):
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        sampler = torch.utils.data.RandomSampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
        self.batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(self.batch_sampler),
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )
        self._iterator = iter(self.loader)

    def __iter__(self):
        return self # 自身即迭代器

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return len(self.batch_sampler)


"""
    数据加载函数，根据数据集名称和其他参数选择合适的数据集类和数据增强方式
"""
mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

def load_data(args, data_index_txt, batch_size, train, infinite_data_loader = False, num_workers=4, use_fixmatch=False, multi_view=False, multi_view_index=None):
    transform = {
        'train': transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)  
            ]),

        'test': transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)  
            ])
    }

    transformation = transform['train' if train else 'test']
    if use_fixmatch:
        transformation = TransformFixMatch(transformation,args)

    if multi_view:
        data = MultiViewDataset(index_txt=data_index_txt, transform=transformation, use_fixmatch=use_fixmatch, views_index=multi_view_index)
    else:
        data = SingleViewDataset(index_txt=data_index_txt, transform=transformation)

    data_loader = get_data_loader(data, batch_size=batch_size, shuffle=True if train else False, drop_last=True if train else False,
                                    num_workers=num_workers, infinite_data_loader=infinite_data_loader)
    n_class = len(data.classes)
    return data_loader, n_class