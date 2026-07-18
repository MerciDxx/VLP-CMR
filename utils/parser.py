import configargparse
from utils.tools import str2bool, str2list





def get_parser():
    parser = configargparse.ArgParser(
        description="VLP-CMR configuration parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    
    # general configuration
    parser.add_argument('--config', is_config_file=True, help='YAML path')
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_info', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True, choices=["MI3DOR", "MI3DOR-2", "GraspNet", "PointDA"])
    parser.add_argument('--use_amp', type=str2bool, default=False)

    # network related
    parser.add_argument('--model_name', type=str, required=True, choices=["RN50", "VIT-B", "RN101", "VIT-L"])

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--tgt_multi_view', type=str2bool, required=True)
    parser.add_argument('--tgt_multi_view_index', type=str2list, default=None, help="Comma-separated list of indices for multi-view target domain, e.g., '0,1,2' or '[0,1,2]'")

    # training related
    parser.add_argument('--l_batch_size', type=int, default=32)
    parser.add_argument('--u_batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="Label smoothing factor for CrossEntropyLoss")
    parser.add_argument("--n_iter_per_epoch", type=int, default=100, help="Used in Iteration-based training")
    parser.add_argument('--early_stop_patience', type=int, default=20, help="Early stopping patience in terms of epochs")

    # view selection related
    parser.add_argument('--mv_select_mode', type=str, default="All", choices=["All", "Soft", "MV_CLIP"])
    parser.add_argument('--top_k', type=int, default=4, help="Number of top views to select for MV_CLIP mode")
    parser.add_argument("--use_mean_text_score", type=str2bool, default=False, help="Use average multiview feat + clip text to compute view selection entropy")
    parser.add_argument('--p_momentum', type=float, default=0.9, help="Momentum for updating the P_source domain prototype memory")
    parser.add_argument('--w_vis', type=float, default=0.5, help="Weight for visual features in the fused representation")

    # FixMatch
    parser.add_argument('--fixmatch', type=str2bool, default=False)
    parser.add_argument('--fixmatch_threshold', type=float, default=0.95)
    parser.add_argument('--fixmatch_factor', type=float, default=0.5)

    # optimizer related
    parser.add_argument('--optim_type', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--multiple_lr_classifier', type=float, default=10)

    # loss related
    parser.add_argument('--lambda1', type=float, default=0.25)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=0.025)
    parser.add_argument('--clf_loss', type=str, default="cross_entropy")

    # learning rate scheduler related
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    # CVCD related
    parser.add_argument('--cvcd', type=str2bool, default=False)
    parser.add_argument('--T_cvcd', type=float, default=2.0)
    parser.add_argument('--lambda_cvcd', type=float, default=0.0)

    return parser.parse_args()
