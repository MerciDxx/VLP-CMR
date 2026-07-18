import torch
import argparse

def cross_performance(final_adj, gtrue_g, gtrue_q):
    """
    纯 PyTorch 实现的 3D 跨域检索指标评估
    参数:
    final_adj: Tensor, shape (Gallery数量, Query数量), 距离越小越相似
    gtrue_g:   Tensor, Gallery 真实标签 (1D Tensor)
    gtrue_q:   Tensor, Query 真实标签 (1D Tensor)
    """
    gallery_num, query_num = final_adj.shape
    device = final_adj.device
    
    # 将标签强转为整型以作索引和匹配
    gtrue_g = gtrue_g.long()
    gtrue_q = gtrue_q.long()
    
    # 1. 距离矩阵排序获取排序索引 (按列升序)
    sort_idx = torch.argsort(final_adj, dim=0) # shape: (gallery_num, query_num)
    
    # 重排 Gallery 标签
    sorted_gallery_labels = gtrue_g[sort_idx]  # shape: (gallery_num, query_num)
    
    # 2. 生成结果匹配矩阵 rresult, 相似的置1，不相似的置0
    # shape转置成: (query_num, gallery_num)
    rresult = (sorted_gallery_labels == gtrue_q.unsqueeze(0)).T.float()
    
    # --- 全局统计 ---
    max_m = gtrue_g.max().item()
    # 统计每个类别的数量，bincount长度自动拉升至最大类别号+1
    number_m = torch.bincount(gtrue_g)
    
    # 计算 PR 曲线系列
    ETH_p = torch.zeros(gallery_num, device=device)
    ETH_rr = torch.zeros(gallery_num, device=device)
    total_relevant_sum = rresult.sum().item()
    
    cumulative_rresult = torch.cumsum(rresult, dim=1) 
    
    for K in range(1, gallery_num + 1):
        s = cumulative_rresult[:, K-1].sum().item()
        ETH_p[K-1] = s / (query_num * K)
        ETH_rr[K-1] = s / total_relevant_sum if total_relevant_sum > 0 else 0
        
    # 【AUC_0】使用积分梯形法则 torch.trapezoid
    AUC_0 = torch.trapezoid(ETH_p, ETH_rr).item()
    
    # 【NN (Rank-1)】
    NN = (rresult[:, 0].sum() / query_num).item()
    
    FT_list, ST_list, DCG_list, NMRR_list = [], [], [], []
    T_max = number_m.max().item()
    
    # 进入核心循环
    for i in range(query_num):
        label_q = gtrue_q[i].item()
        C = number_m[label_q].item() # 查询类别的同类总数
        
        if C == 0:
            continue
            
        # 【FT 和 ST】
        FT_list.append((rresult[i, :C].sum() / C).item())
        ST_list.append((rresult[i, :2*C - 1].sum() / C).item())
        
        # 【DCG】(特化逻辑：从 Rank 2 往后算)
        DCG_k = rresult[i, 1].item() if gallery_num > 1 else 0
        DCG_data = 1.0
        if C > 2:
            ranks = torch.arange(3, C + 1, dtype=torch.float32, device=device)
            weights = torch.log2(ranks - 1)
            DCG_k += (rresult[i, 2:C] / weights).sum().item()
            DCG_data += (1.0 / weights).sum().item()
        DCG_list.append(DCG_k / DCG_data if DCG_data > 0 else 0)
        
        # 【ANMRR】
        S_k = min(4 * C, 2 * T_max)
        r = torch.zeros(C, device=device)
        for k in range(1, C + 1):
            if rresult[i, k - 1].item() == 1.0:
                r[k - 1] = k
            else:
                r[k - 1] = S_k + 1
                
        nmrr = ((r.sum().item() / C) - C / 2 - 0.5) / (S_k - C / 2 + 0.5)
        NMRR_list.append(nmrr)

    FT = sum(FT_list) / len(FT_list) if FT_list else 0
    ST = sum(ST_list) / len(ST_list) if ST_list else 0
    DCG = sum(DCG_list) / len(DCG_list) if DCG_list else 0
    ANMRR = sum(NMRR_list) / len(NMRR_list) if NMRR_list else 0
    
    # 【F-Measure @ 20】
    idx_20 = min(19, gallery_num - 1)
    s_20 = cumulative_rresult[:, idx_20].sum().item()
    P = s_20 / (query_num * 20)
    R = s_20 / total_relevant_sum if total_relevant_sum > 0 else 0
    F_measure = 2 / ((1/P) + (1/R)) if (P + R) > 0 else 0
    
    return [NN, FT, ST, F_measure, DCG, ANMRR, AUC_0]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="跨模态/跨域检索指标评测脚本")
    
    # 添加 source_file 参数，并指定当前的路径作为默认值
    parser.add_argument(
        '--source_file', 
        type=str, 
        help="源域特征文件 (source_test_fea.pth) 的绝对或相对路径"
    )
    
    # 添加 target_file 参数，同样指定默认值
    parser.add_argument(
        '--target_file', 
        type=str, 
        help="目标域特征文件 (model_test_012_fea.pth) 的绝对或相对路径"
    )

    parser.add_argument('--gpu_id', type=int, default=0, help="指定使用的GPU编号 (0-7)")
    
    return parser.parse_args()


def run_evaluation(args):
    # 为了加速，如果有GPU可以使用cuda
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 从解析好的 args 中获取路径
    source_file = args.source_file
    target_file = args.target_file
    
    print(f"Loading source from: {source_file}")
    print(f"Loading target from: {target_file}")

    source_data = torch.load(source_file, map_location=device, weights_only=False)
    target_data = torch.load(target_file, map_location=device, weights_only=False)
    
    source_features = source_data['fea'].to(device)
    source_labels = source_data['label'].flatten().to(device) # 打平保证是一维张量
    
    ttf = target_data['fea'].to(device)
    ttl = target_data['label'].flatten().to(device)
    
    # 归一化
    qf = torch.nn.functional.normalize(source_features, p=2, dim=1)
    gf = torch.nn.functional.normalize(ttf, p=2, dim=1)
    # 余弦距离
    final_adj = 1 - torch.mm(qf, gf.t())
    
    # 4. 指标评测调用
    # 矩阵转置 final_adj.T 转变为 (Gallery数量, Query数量) 的输入形状以贴合内部计算要求
    pingce = cross_performance(final_adj.T, ttl, source_labels)
    
    # 5. 打印各项指标
    labels = ["NN", "FT", "ST", "F-Measure", "DCG", "ANMRR", "AUC_0"]
    for label, val in zip(labels, pingce):
        print(f"{label:<10}: {val:.4f}")


if __name__ == '__main__':
    # 先解析参数，再传入主函数
    args = parse_args()
    run_evaluation(args)