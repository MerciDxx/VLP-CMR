import argparse
import itertools
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ==================== 配置区域 ====================
# 参数网格配置（按顺序：optimizer, lr, weight_decay, momentum, lambda1, lambda2, lambda3）
PARAM_GRID = {
    "optimizer": ["SGD"],
    "lr": [3e-6],
    "weight_decay": [5e-4],
    "momentum": [0.9],      # 仅SGD使用
    "lambda1": [0.25, 0.5, 1.0, 2.0, 3.0],
    "lambda2": [0.1, 0.2, 0.5, 1.0],
    "lambda3": [0.0, 0.025],
}

# 数据集对配置（enabled=False 可跳过）
DATASET_PAIRS = [
    {"name": "M2S_star",  "src_domain": "modelnet", "tgt_domain": "scannet",  "enabled": False},
    {"name": "M2S",       "src_domain": "modelnet", "tgt_domain": "shapenet", "enabled": True},
    {"name": "S_star2M",  "src_domain": "scannet",  "tgt_domain": "modelnet", "enabled": True},
    {"name": "S_star2S",  "src_domain": "scannet",  "tgt_domain": "shapenet", "enabled": True},
    {"name": "S2M",       "src_domain": "shapenet", "tgt_domain": "modelnet", "enabled": True},
    {"name": "S2S_star",  "src_domain": "shapenet", "tgt_domain": "scannet",  "enabled": True},
]

# 网格搜索模式（True=跑所有参数组合，False=只跑下面的单组参数）
GRID_SEARCH_MODE = True

SINGLE_RUN_PARAMS = {
    "optimizer": "SGD",
    "lr": 3e-6,
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "lambda1": 0.25,
    "lambda2": 0.1,
    "lambda3": 0.025,
}
# =================================================


def print_msg(msg):
    """带时间戳的打印，自动flush"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def _generate_param_combinations():
    """生成所有参数组合"""
    if not GRID_SEARCH_MODE:
        return [SINGLE_RUN_PARAMS]

    optimizers = PARAM_GRID["optimizer"]
    other_keys = [k for k in PARAM_GRID if k != "optimizer"]
    other_values = [PARAM_GRID[k] for k in other_keys]

    combos = []
    for opt in optimizers:
        for values in itertools.product(*other_values):
            combo = {"optimizer": opt}
            for k, v in zip(other_keys, values):
                if k == "momentum" and opt != "SGD":
                    continue
                combo[k] = v
            combos.append(combo)
    return combos


def _params_to_string(params):
    """参数字典 -> 文件名用字符串，如 SGD_3e-06_0.0005_0.9_2.0_1.0_0.2"""
    def fmt(v):
        if isinstance(v, float):
            if abs(v) < 0.01 and v != 0:
                s = f"{v:.0e}".replace("e-0", "e-").replace("e-", "e-").replace("e+0", "e")
            else:
                s = f"{v:.6f}".rstrip('0').rstrip('.')
            return s
        return str(v)

    parts = [str(params["optimizer"])]
    parts.append(fmt(params["lr"]))
    parts.append(fmt(params.get("weight_decay", 5e-4)))
    if params["optimizer"] == "SGD" and "momentum" in params:
        parts.append(fmt(params["momentum"]))
    for k in ["lambda1", "lambda2", "lambda3"]:
        if k in params:
            parts.append(fmt(params[k]))
    return "_".join(parts)


def build_task_list(config_path):
    """生成任务列表：启用的数据集对 × 参数组合"""
    tasks = []
    params_list = _generate_param_combinations()

    for pair in DATASET_PAIRS:
        if not pair.get("enabled", True):
            continue
        for p in params_list:
            param_str = _params_to_string(p)
            tasks.append({
                "name": f"{pair['name']}_{param_str}",
                "src_domain": pair["src_domain"],
                "tgt_domain": pair["tgt_domain"],
                "params": p,
            })
    return tasks


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


def query_gpu_status():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError:
        print_msg("WARNING: nvidia-smi 查询失败，30秒后重试...")
        time.sleep(30)
        return []
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, mem_used, util = parts
        gpus.append({
            "index": int(idx),
            "memory_used": int(mem_used),
            "utilization": int(util),
        })
    return gpus


def is_gpu_free(gpu_info, mem_threshold, util_threshold):
    return (
        gpu_info["memory_used"] < mem_threshold
        and gpu_info["utilization"] < util_threshold
    )


def make_log_info(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def build_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def run_command(cmd, cwd, output_path, logger):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\n==== CMD: {' '.join(cmd)} ===\n")
        f.flush()
        result = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=f)
    if result.returncode != 0:
        logger.error("Command failed: %s", " ".join(cmd))
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_task(task, task_id, task_total, gpu_id, config_path, output_log_root):
    src_domain = task["src_domain"]
    tgt_domain = task["tgt_domain"]
    params = task.get("params", {})

    log_info = make_log_info(task["name"])
    output_dir = os.path.join(output_log_root, f"{src_domain}2{tgt_domain}", log_info)
    os.makedirs(output_dir, exist_ok=True)
    run_log_path = os.path.join(output_dir, "auto_run.log")
    logger = build_logger(f"auto_run.{log_info}", run_log_path)
    logger.info(
        "Start task %d/%d on GPU %d | name=%s | src_domain=%s | tgt_domain=%s",
        task_id, task_total, gpu_id, task["name"], src_domain, tgt_domain
    )
    logger.info("Params: %s", params)

    train_cmd = [
        "python", "main.py",
        "--config", config_path,
        "--gpu_id", str(gpu_id),
        "--log_info", log_info,
        "--src_domain", src_domain,
        "--tgt_domain", tgt_domain,
    ]

    if params:
        train_cmd.extend(["--optimizer", params["optimizer"]])
        train_cmd.extend(["--lr", str(params["lr"])])
        train_cmd.extend(["--weight_decay", str(params.get("weight_decay", 5e-4))])
        if params["optimizer"] == "SGD" and "momentum" in params:
            train_cmd.extend(["--momentum", str(params["momentum"])])
        for k in ["lambda1", "lambda2", "lambda3"]:
            if k in params:
                train_cmd.extend([f"--{k}", str(params[k])])

    train_log = os.path.join(output_dir, "train.log")
    logger.info("Train command -> %s", train_log)

    print_msg(f"[{task_id}/{task_total}] START  | {task['name']} | GPU:{gpu_id}")

    try:
        run_command(train_cmd, REPO_ROOT, train_log, logger)
        print_msg(f"[{task_id}/{task_total}] DONE   | {task['name']} | GPU:{gpu_id}")
    except RuntimeError:
        print_msg(f"[{task_id}/{task_total}] FAILED | {task['name']} | GPU:{gpu_id}")
        # 创建失败标记文件
        with open(os.path.join(output_dir, "FAILED"), "w") as f:
            f.write(f"Task failed at {datetime.now()}\n")
        # 不往上抛，让调度器继续跑其他任务


def main():
    parser = argparse.ArgumentParser(description="Auto scheduler for VLP-CMR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mem-threshold", type=int, default=1000)
    parser.add_argument("--util-threshold", type=int, default=3)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="仅打印任务列表，不执行训练")
    args = parser.parse_args()

    config_path = args.config
    config = parse_simple_yaml(config_path)

    output_log_root = os.path.join(REPO_ROOT, "output_log", config.get("datasets"))
    os.makedirs(output_log_root, exist_ok=True)
    tasks = build_task_list(args.config)

    # ===== 打印任务摘要 =====
    print_msg("=" * 60)
    print_msg("GRID SEARCH SCHEDULER")
    print_msg(f"Config:     {config_path}")
    print_msg(f"Grid mode:  {'ON' if GRID_SEARCH_MODE else 'OFF'}")
    print_msg(f"Total tasks: {len(tasks)}")
    print_msg(f"GPU threshold: mem<{args.mem_threshold}MB, util<{args.util_threshold}%")
    print_msg("-" * 60)
    for i, t in enumerate(tasks, 1):
        print_msg(f"  {i:4d}. {t['name']}")
    print_msg("=" * 60)

    if args.dry_run:
        print_msg("[DRY-RUN] 仅显示任务列表，退出。")
        return

    # ===== 主循环 =====
    pending = list(tasks)
    running = {}
    task_total = len(tasks)
    done_count = 0
    fail_count = 0

    print_msg("开始执行任务队列...")

    while pending or running:
        # 清理已完成的 worker
        finished = []
        for gpu_id, worker in running.items():
            if not worker.is_alive():
                worker.join()
                finished.append(gpu_id)
        for gpu_id in finished:
            running.pop(gpu_id, None)

        # 统计完成/失败（检查输出目录下的 FAILED 文件）
        # 这里简单用计数器，run_task 里不抛异常就是 done，抛了就是 fail
        # 但由于 run_task 现在吞掉了异常，我们在回调里没法区分。改用更简单的方式：
        # 直接在 finished 时不区分，进度只显示 pending 和 running 数量

        if pending:
            gpus = query_gpu_status()
            free_gpus = [
                gpu for gpu in gpus
                if is_gpu_free(gpu, args.mem_threshold, args.util_threshold)
            ]
            if len(free_gpus) <= 1:
                # 不刷屏，只在有变化时打印
                time.sleep(args.poll_interval)
                continue

            free_gpus_sorted = sorted(free_gpus, key=lambda g: g["index"], reverse=True)
            for gpu in free_gpus_sorted:
                gpu_id = gpu["index"]
                if gpu_id in running:
                    continue
                if not pending:
                    break
                task = pending.pop(0)
                task_id = task_total - len(pending)
                worker = threading.Thread(
                    target=run_task,
                    args=(task, task_id, task_total, gpu_id, config_path, output_log_root),
                    daemon=True,
                )
                running[gpu_id] = worker
                worker.start()

        # 定期打印进度
        # print_msg(f"STATUS | pending:{len(pending)} running:{len(running)} finished:{task_total - len(pending) - len(running)}")

        time.sleep(args.poll_interval)

    print_msg("=" * 60)
    print_msg("ALL TASKS COMPLETED.")
    print_msg("=" * 60)


if __name__ == "__main__":
    main()