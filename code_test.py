import utils.datasets as ds

pa = ds.Scannet('/data1/dengxuxiang/datasets/PointDA_data/scannet', 'test')

pc, label = pa[0]
print(type(label))
print(label)

print("=====================================")

pa = ds.PointDA('/data1/dengxuxiang/datasets/PointDA_data/modelnet', 'test')

pc, label = pa[0]
print(type(label))
print(label)

print("=====================================")

pa = ds.PointDA('/data1/dengxuxiang/datasets/PointDA_data/shapenet', 'test')

pc, label = pa[0]
print(type(label))
print(label)