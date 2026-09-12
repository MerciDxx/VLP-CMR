import argparse
import logging
import os
import subprocess
import threading
import time
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))



"""
	这里定义不同的任务区别
"""
def build_task_list(config_path):
    filename = os.path.basename(config_path)
    params_suffix = os.path.splitext(filename)[0]

    tasks = []


    tasks.append({
        "name": f"M2S_star_{params_suffix}",
		"src_domain": "modelnet",
        "tgt_domain": "scannet",
    })

    tasks.append({
        "name": f"M2S_{params_suffix}",
		"src_domain": "modelnet",
        "tgt_domain": "shapenet",
    })



    tasks.append({
        "name": f"S_star2M_{params_suffix}",
		"src_domain": "scannet",
        "tgt_domain": "modelnet",
    })

    tasks.append({
        "name": f"S_star2S_{params_suffix}",
		"src_domain": "scannet",
        "tgt_domain": "shapenet",
    })



    tasks.append({
        "name": f"S2M_{params_suffix}",
		"src_domain": "shapenet",
        "tgt_domain": "modelnet",
    })

    tasks.append({
        "name": f"S2S_star_{params_suffix}",
		"src_domain": "shapenet",
        "tgt_domain": "scannet",
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


def query_gpu_status():
	cmd = [
		"nvidia-smi",
		"--query-gpu=index,memory.used,utilization.gpu",
		"--format=csv,noheader,nounits",
	]
	output = subprocess.check_output(cmd, text=True)
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

	log_info = make_log_info(task["name"])
	output_dir = os.path.join(output_log_root, f"{src_domain}2{tgt_domain}", log_info)
	os.makedirs(output_dir, exist_ok=True)
	run_log_path = os.path.join(output_dir, "auto_run.log")
	logger = build_logger(f"auto_run.{log_info}", run_log_path)
	logger.info(
		"Start task %d/%d on GPU %d | name=%s | src_domain=%s | tgt_domain=%s",
		task_id,
		task_total,
		gpu_id,
		task["name"],
		src_domain,
		tgt_domain
	)

	train_cmd = [
		"python",
		"main.py",
		"--config",
		config_path,
		"--gpu_id",
		str(gpu_id),
		"--log_info",
		log_info,
		"--src_domain",
        src_domain,
		"--tgt_domain",
		tgt_domain,
	]

	train_log = os.path.join(output_dir, "train.log")
	logger.info("Train command -> %s", train_log)
	run_command(train_cmd, REPO_ROOT, train_log, logger)



def main():
	parser = argparse.ArgumentParser(description="Auto scheduler for VLP-CMR")
	parser.add_argument("--config", type=str, required=True)
	parser.add_argument("--mem-threshold", type=int, default=1000)
	parser.add_argument("--util-threshold", type=int, default=3)
	parser.add_argument("--poll-interval", type=int, default=60)
	args = parser.parse_args()

	config_path = args.config
	config = parse_simple_yaml(config_path)
	
	output_log_root = os.path.join(REPO_ROOT, "output_log", config.get("datasets"))
	os.makedirs(output_log_root, exist_ok=True)
	tasks = build_task_list(args.config)
	pending = list(tasks)
	running = {}
	task_total = len(tasks)

	while pending or running:
		# Cleanup finished workers
		finished = []
		for gpu_id, worker in running.items():
			if not worker.is_alive():
				finished.append(gpu_id)
		for gpu_id in finished:
			running.pop(gpu_id, None)

		if pending:
			gpus = query_gpu_status()
			free_gpus = [
				gpu for gpu in gpus
				if is_gpu_free(gpu, args.mem_threshold, args.util_threshold)
			]
			if len(free_gpus) <= 0:
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

		time.sleep(args.poll_interval)


if __name__ == "__main__":
	main()
