import os
import shutil

# 基础路径
BASE = "/data1/dengxuxiang/VLP-CMR/output_log/PointDA"
os.chdir(BASE)

# 迁移规则列表：(前缀, 排除前缀(空则不排除), 目标文件夹)
rules = [
    ("M2S", "M2S_star", "modelnet2shapenet"),
    ("M2S_star", "", "modelnet2scannet"),
    ("S2M", "S2M_star", "shapenet2modelnet"),
    ("S2S_star", "", "shapenet2scannet"),
    ("S_star2M", "", "scannet2modelnet"),
    ("S_star2S", "", "scannet2shapenet"),
]

# 遍历当前目录所有文件夹
all_items = [d for d in os.listdir() if os.path.isdir(d)]

for prefix, exclude, target in rules:
    print(f"\n=== 处理 {prefix}*  -> {target} ===")
    for item in all_items:
        # 跳过目标文件夹本身
        if item == target:
            continue
        # 匹配前缀
        if item.startswith(prefix):
            # 需要排除的情况
            if exclude and item.startswith(exclude):
                continue
            # 执行移动
            src_path = os.path.join(BASE, item)
            dst_path = os.path.join(BASE, target, item)
            shutil.move(src_path, dst_path)
            print(f"Moved: {item} --> {target}/")

print("\n所有目录迁移完毕")