import torch
import os
import argparse
from tqdm import tqdm

from models.TransferNet import TransferNet
from utils import data_loader
from utils.tools import str2bool, str2list



def extract_and_save_features(args, training_config:dict):
    if args.gpu_id >= 0:
        gpu_id = args.gpu_id
    else:
        gpu_id = training_config.get('gpu_id')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    target_test_loader, _ = data_loader.load_data(
        args, args.txt_path, args.batch_size, train=False, infinite_data_loader=False, num_workers=training_config.get('num_workers'), 
        multi_view=args.tgt_multi_view, multi_view_index=args.tgt_multi_view_index)

    class DummyArgs:
        pass
    model_args = DummyArgs()
    model_args.model_name = training_config.get('model_name')
    model_args.num_class = training_config.get('num_class')
    model_args.device = device
    model_args.fixmatch = False
    model_args.datasets = training_config.get('datasets')
    model_args.mv_select_mode = training_config.get('mv_select_mode')  # 需要命令行新增参数
    model_args.top_k = training_config.get('top_k')
    model_args.use_mean_text_score = training_config.get('use_mean_text_score')  # 需要命令行新增参数
    model_args.p_momentum = training_config.get('p_momentum')
    model_args.w_vis = training_config.get('w_vis')

    # 1.  
    model = TransferNet(model_args, train=False).to(device)
    
    # 2. 严格使用 vlp-uda 的解析逻辑加载权重
    print(f"Loading weights from {args.model_path}...")
    checkpoints = torch.load(args.model_path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoints["full_model_state_dict"])
    print("Successfully loaded dense weights.")

    # 3. 将加载好权重的模型推至GPU并开启验证模式
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    print("Start extracting features...")
    with torch.no_grad():
        for imgs, labels in tqdm(target_test_loader, desc="Extracting"):
            if len(imgs.shape) == 5:  # (B, V, C, H, W)
                imgs = imgs.to(device)
                B, V, C, H, W = imgs.shape
                imgs_flat = imgs.view(B * V, C, H, W)
                features_flat = model.base_network.forward_features(imgs_flat)
                features_multiviews = features_flat.view(B, V, -1)
                # 关键：调用多视角选择模块，和训练推理保持一致
                _, features_multiviews = model.multi_views_selection(features_multiviews)
                pooled_features = model.target_feature_pooling(features_multiviews)
                
                all_features.append(pooled_features.cpu())
                all_labels.append(labels.cpu())
            else:
                imgs = imgs.to(device)
                features = model.base_network.forward_features(imgs)
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0) 
    all_labels = torch.cat(all_labels, dim=0)     

    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    save_dict = {
        'fea': all_features,
        'label': all_labels
    }
    torch.save(save_dict, args.output_path)
    print(f"Successfully saved to {args.output_path}")





def parse_simple_yaml(path):
	config = {}
	with open(path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			if ":" not in line:
				continue
			key, value = line.split(":", 1)
			key = key.strip()
			value = value.strip()
			if value.startswith("\"") and value.endswith("\""):
				value = value[1:-1]
			if value.startswith("'") and value.endswith("'"):
				value = value[1:-1]
			# Basic type parsing
			if value.lower() in ("true", "false"):
				value = value.lower() == "true"
			else:
				try:
					if "." in value:
						value = float(value)
					else:
						value = int(value)
				except ValueError:
					pass
			config[key] = value
	return config




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract features for VLP-UDA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--txt_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tgt_multi_view', type=str2bool, required=True)
    parser.add_argument('--tgt_multi_view_index', type=str2list, default=None, help="Comma-separated list of indices for multi-view target domain, e.g., '0,1,2' or '[0,1,2]'") # 有了
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu_id', type=int, default=-1)

    args = parser.parse_args()
    training_config = parse_simple_yaml(args.config)
    extract_and_save_features(args, training_config)