from utils.datasets import PointDA, Scannet

PointDA_root = "/data1/dengxuxiang/datasets/PointDA_data"

modelnet_src = PointDA(f'{PointDA_root}/modelnet', split='train')
# shapenet_src = PointDA(f'{PointDA_root}/shapenet', split='train')
scannet_src = Scannet(f'{PointDA_root}/scannet', split='train')
# modelnet_tar = PointDA(f'{PointDA_root}/modelnet', split='test')
# shapenet_tar = PointDA(f'{PointDA_root}/shapenet', split='test')
# scannet_tar = Scannet(f'{PointDA_root}/scannet', split='test')


print(modelnet_src.class_to_idx)
print(modelnet_src.idx_to_class)
print("-----------------------------")
print(scannet_src.class_to_idx)
print(scannet_src.idx_to_class)

sample = modelnet_src[0]
print(sample)

print("-----------------------------")

import numpy
npdata = numpy.load("/data1/dengxuxiang/datasets/PointDA_data/modelnet/bathtub/train/bathtub_0001.npy")



