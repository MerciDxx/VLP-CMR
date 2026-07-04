import torch

class Arg:
    def __init__(self):
        self.num_class = 10
        self.device = 'cuda'
        self.label_smoothing = 0.1
        self.model_name = 'VIT-B'
        self.datasets = 'MI3DOR'
        self.max_iter = 10000
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.scheduler = False
        self.lr_gamma = 0.1
        self.lr_decay = 0.75
        # Add other necessary attributes here

args = Arg()

def get_optimizer(model, args):
    # 原方法使用SGD优化器 我建议之后尝试采用adam优化器
    initial_lr = args.lr if not args.scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)

    optimizer = torch.optim.SGD(
        params, 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay, 
        nesterov=True
    )
    return optimizer
    

def get_lr_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  (args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)))
    return scheduler


import matplotlib.pyplot as plt
from torch import nn

# 假设你有一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)
        
    def get_parameters(self, initial_lr=0.001):
        # 模拟原始代码中的参数分组
        return [
            {'params': self.fc.weight, 'lr': initial_lr},
            {'params': self.fc.bias, 'lr': initial_lr * 1000}  # 可以有不同的学习率
        ]
    
    def forward(self, x):
        return self.fc(x)

def test_lr_scheduler():
    """测试学习率调度器的变化情况"""
    
    # 设置参数
    args = Arg()
    args.scheduler = True  # 启用调度器
    args.lr = 3e-6
    args.lr_gamma = 0.0003
    args.lr_decay = 0.75
    args.max_iter = 10000
    
    # 创建模型和优化器
    model = SimpleModel()
    optimizer = get_optimizer(model, args)
    scheduler = get_lr_scheduler(optimizer, args)
    
    # 记录学习率变化
    lrs = []
    param_lrs = {'weight': [], 'bias': []}
    
    print("=" * 60)
    print("学习率变化跟踪 (前20步)")
    print("=" * 60)
    print(f"{'Step':<10} {'Weight LR':<15} {'Bias LR':<15} {'Lambda':<10}")
    print("-" * 60)
    
    for step in range(args.max_iter):
        # 记录当前学习率
        current_lrs = []
        for param_group in optimizer.param_groups:
            current_lrs.append(param_group['lr'])
        
        lrs.append(optimizer.param_groups[0]['lr'])
        param_lrs['weight'].append(optimizer.param_groups[0]['lr'])
        param_lrs['bias'].append(optimizer.param_groups[1]['lr'])
        
        # 打印前20步
        if step < 20:
            lambda_val = (args.lr * (1. + args.lr_gamma * step) ** (-args.lr_decay))
            print(f"{step:<10} {optimizer.param_groups[0]['lr']:<15.8f} "
                  f"{optimizer.param_groups[1]['lr']:<15.8f} {lambda_val:<10.6f}")
        
        # 模拟训练步骤并更新学习率
        scheduler.step()
    
    # 打印关键节点的学习率
    print("\n" + "=" * 60)
    print("关键节点学习率")
    print("=" * 60)
    key_steps = [0, 100, 500, 1000, 2000, 5000, 9999]
    for step in key_steps:
        print(f"Step {step:>5}: Weight LR = {param_lrs['weight'][step]:.8f}, "
              f"Bias LR = {param_lrs['bias'][step]:.8f}")
    
    # 可视化学习率变化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图1: 完整的学习率变化曲线
    steps = range(args.max_iter)
    ax1.plot(steps, param_lrs['weight'], label='Weight LR', linewidth=2)
    ax1.plot(steps, param_lrs['bias'], label='Bias LR', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title(f'Learning Rate Schedule (γ={args.lr_gamma}, decay={args.lr_decay})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 对数尺度的学习率变化
    ax2.semilogy(steps, param_lrs['weight'], label='Weight LR', linewidth=2)
    ax2.semilogy(steps, param_lrs['bias'], label='Bias LR', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Learning Rate (log scale)', fontsize=12)
    ax2.set_title('Learning Rate Schedule (Log Scale)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("lr_schedule.png", dpi=300, bbox_inches='tight')
    
    # 分析学习率变化特点
    print("\n" + "=" * 60)
    print("学习率变化分析")
    print("=" * 60)
    print(f"初始学习率: {param_lrs['weight'][0]:.8f}")
    print(f"最终学习率: {param_lrs['weight'][-1]:.8f}")
    print(f"学习率衰减倍数: {param_lrs['weight'][0] / param_lrs['weight'][-1]:.2f}x")
    print(f"前1000步衰减率: {(1 - param_lrs['weight'][1000]/param_lrs['weight'][0])*100:.2f}%")
    
    # 比较两种参数的学习率比例
    print(f"\nWeight和Bias的学习率比例始终保持: {param_lrs['bias'][0]/param_lrs['weight'][0]:.1f}x")
    
    return lrs, param_lrs

def compare_with_and_without_scheduler():
    """对比有无调度器的情况"""
    
    args = Arg()
    
    # 情况1: 有调度器
    args.scheduler = True
    model1 = SimpleModel()
    optimizer1 = get_optimizer(model1, args)
    scheduler1 = get_lr_scheduler(optimizer1, args)
    
    # 情况2: 无调度器（固定学习率）
    args.scheduler = False
    model2 = SimpleModel()
    optimizer2 = get_optimizer(model2, args)
    
    lrs_with_scheduler = []
    lrs_without_scheduler = []
    
    for step in range(1000):
        lrs_with_scheduler.append(optimizer1.param_groups[0]['lr'])
        lrs_without_scheduler.append(optimizer2.param_groups[0]['lr'])
        scheduler1.step()
    
    # 可视化对比
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lrs_with_scheduler, label='With Scheduler', linewidth=2)
    ax.plot(lrs_without_scheduler, label='Without Scheduler (Fixed)', linewidth=2, linestyle='--')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Comparison: With vs Without Scheduler', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
    
    print("\n对比结果:")
    print(f"有调度器 - 初始LR: {lrs_with_scheduler[0]:.6f}, 最终LR: {lrs_with_scheduler[-1]:.6f}")
    print(f"无调度器 - 初始LR: {lrs_without_scheduler[0]:.6f}, 最终LR: {lrs_without_scheduler[-1]:.6f}")

if __name__ == "__main__":
    # 运行主测试
    lrs, param_lrs = test_lr_scheduler()
    
    # 运行对比测试
    compare_with_and_without_scheduler()