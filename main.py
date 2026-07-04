from utils.parser import get_parser
import os
args = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import yaml
import time
import torch
import random
import numpy as np
from utils import data_loader
from models.TransferNet import TransferNet
import logging
from tqdm import tqdm
from utils.tools import AverageMeter, save_model, fix_bn
scaler = torch.amp.GradScaler('cuda')       # 用于混合精度训练的梯度缩放器，帮助稳定训练过程并防止数值下溢
from utils.render import Realistic_Projection       # 用于GraspNet数据集的点云投影工具，将点云数据转换为图像形式，以便输入到CLIP模型中进行特征提取


def set_random_seed(seed):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


pc_views = Realistic_Projection()
def real_proj(pc, imsize=224):
    img = pc_views.get_img(pc.float()).to(pc.device)
    img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
    return img


"""
    加载源域和目标域的数据，返回数据加载器和类别数
"""
def load_data(args):
    if args.datasets in ["GraspNet", "PointDA"]:
        return data_loader.load_PointCloud_data(args)

    use_fixmatch = args.fixmatch
    index_txt_src = os.path.join(args.data_dir, args.src_domain)
    index_txt_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        args, index_txt_src, args.l_batch_size, train=True, infinite_data_loader=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        args, index_txt_tgt, args.u_batch_size, train=True, infinite_data_loader=True, num_workers=args.num_workers, use_fixmatch=use_fixmatch, multi_view=args.tgt_multi_view, multi_view_index=args.tgt_multi_view_index)
    target_test_loader, _ = data_loader.load_data(
        args, index_txt_tgt, args.u_batch_size, train=False, infinite_data_loader=False, num_workers=args.num_workers, multi_view=args.tgt_multi_view, multi_view_index=args.tgt_multi_view_index)
    return source_loader, target_train_loader, target_test_loader, n_class



"""
    获取优化器，使用SGD优化器，并根据参数设置学习率、动量和权重衰减
"""
def get_optimizer(model, args):
    # 原方法使用SGD优化器 我建议之后尝试采用adam优化器
    initial_lr = args.lr if not args.scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)

    if args.optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            params, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay, 
            nesterov=True
        )
        return optimizer
    
    elif args.optim_type == 'Adam':
        # Adam 优化器不需要 momentum 和 nesterov
        optimizer = torch.optim.Adam(
            params, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        return optimizer

    elif args.optim_type == 'AdamW':
        # 【强烈推荐】微调专用 AdamW，权重衰减（weight decay）更准确
        optimizer = torch.optim.AdamW(
            params, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        return optimizer
        
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optim_type}")




def get_lr_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  (args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)))
    return scheduler



def test(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    first_test = True
    with torch.no_grad():
        for data, target in tqdm(iterable=target_test_loader,desc="Testing..."):
            data, target = data.to(args.device), target.to(args.device)

            if args.datasets in ["GraspNet", "PointDA"]:
                data_images = real_proj(data)
                data = data_images.reshape(data.size(0), 10, data_images.shape[-3], data_images.shape[-2], data_images.shape[-1])

            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            if first_test:
                all_pred = pred
                all_label = target
                first_test = False
            else:
                all_pred = torch.cat((all_pred, pred), 0)
                all_label = torch.cat((all_label, target), 0)

    acc = torch.sum(torch.squeeze(all_pred).float() == all_label) / float(all_label.size()[0]) * 100
    return acc, test_loss.avg



def train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args):
    logging.basicConfig(filename=os.path.join(args.log_dir,'training.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    n_batch = args.n_iter_per_epoch
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    early_stop_counter = 0      # 用于早停的计数器，记录连续多少个epoch没有提升验证集性能
    for e in range(1, args.n_epoch+1):
        model.train()
        model.base_network.apply(fix_bn) # 冻结clip的BN层

        train_loss_clf = AverageMeter()
        train_loss_transfer = AverageMeter()
        train_loss_total = AverageMeter()

        for _ in tqdm(iterable=range(n_batch),desc=f"Train:[{e}/{args.n_epoch}]"):
            optimizer.zero_grad()       # 每个batch开始前都要清零梯度，否则会累积之前的梯度
            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)

            data_target_strong = None       # 用于fixmatch
            if args.fixmatch:
                data_target, data_target_strong = data_target[0], data_target[1]
                data_target, data_target_strong = data_target.to(args.device), data_target_strong.to(args.device)
            else:
                data_target = data_target.to(args.device)

            # 注意，目前点云数据集仅支持非fixmatch
            if args.datasets in ["GraspNet", "PointDA"]:
                source_B = data_source.size(0)
                target_B = data_target.size(0)

                with torch.no_grad():
                    data_source_images = real_proj(data_source)
                    data_source = data_source_images.reshape(source_B, 10, data_source_images.shape[-3], data_source_images.shape[-2], data_source_images.shape[-1])

                    data_target_images = real_proj(data_target)
                    data_target = data_target_images.reshape(target_B, 10, data_target_images.shape[-3], data_target_images.shape[-2], data_target_images.shape[-1])

            if args.use_amp:        # 混合精度训练
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    clf_loss, transfer_loss = model(data_source, label_source, data_target, data_target_strong)
                    loss = clf_loss + transfer_loss
                scaler.scale(loss).backward()           # 反向传播时使用梯度缩放器来处理总损失
                scaler.step(optimizer)                  # 更新参数时使用梯度缩放器
                scaler.update()                         # 更新梯度缩放器的状态
            else:
                clf_loss, transfer_loss = model(data_source, label_source, data_target, data_target_strong)
                loss = clf_loss + transfer_loss
                loss.backward()       # 反向传播计算梯度
                optimizer.step()          # 更新参数

            if scheduler is not None:
                scheduler.step()

            # training loss update
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
        
        # Test 阶段
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)

        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)

        if best_acc < test_acc:
            best_acc = test_acc
            save_model(model,args)
            early_stop_counter = 0      
        else:
            early_stop_counter += 1     

        logging.info(info)
        tqdm.write(info)
        time.sleep(1)

        if early_stop_counter >= args.early_stop_patience:
            tqdm.write(f"Early stopping triggered after {e} epochs without improvement.")
            break

    tqdm.write('Transfer result: {:.4f}'.format(best_acc))



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
    setattr(args, "device", torch.device('cuda'))
    set_random_seed(args.seed)

    source_loader, target_train_loader, target_test_loader, num_class = load_data(args)
    setattr(args, "num_class", num_class)
    setattr(args, "max_iter", 10000)

    log_dir = f'log/{args.model_name}/{args.datasets}/{args.src_domain}2{args.tgt_domain}/{args.log_info}/'
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            print(f"Error creating directory {log_dir}: {e}")
            exit(1)
    setattr(args, "log_dir", log_dir)

    print(args)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    model = TransferNet(args).to(args.device)
    print(model)
    optimizer = get_optimizer(model, args)

    if args.scheduler:
        scheduler = get_lr_scheduler(optimizer,args)
    else:
        scheduler = None
    print(f"Base Network: {args.model_name}")
    print(f"Source Domain: {args.src_domain}")
    print(f"Target Domain: {args.tgt_domain}")
    print(f"FixMatch: {args.fixmatch}")
    
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)



if __name__ == "__main__":
    main()