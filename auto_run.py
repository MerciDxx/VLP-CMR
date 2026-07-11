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
def build_task_list():
    tasks = []

    tasks.append({
        "name": "MV_CLIP",
        "tgt_domain": "Model_net_40_target_train.txt",
        "tgt_multi_view": True,
        "tgt_multi_view_index": None,
        "test_tgt_txt": "Model_net_40_target_test.txt",
        "test_tgt_multi_view": True,
        "test_tgt_multi_view_index": None,
		"mv_select_mode": "MV_CLIP",
		"top_k": 4
    })

    tasks.append({
        "name": "Soft",
        "tgt_domain": "Model_net_40_target_train.txt",
        "tgt_multi_view": True,
        "tgt_multi_view_index": None,
        "test_tgt_txt": "Model_net_40_target_test.txt",
        "test_tgt_multi_view": True,
        "test_tgt_multi_view_index": None,
		"mv_select_mode": "Soft",
		"top_k": 4
    })

	# # 选择NN得分靠前的4个视角
    # tasks.append({
    #     "name": "viwes[2,3,6,12]_with_fixmatch",
    #     "tgt_domain": "target_train.txt",
    #     "tgt_multi_view": True,
    #     "tgt_multi_view_index": [1, 2, 5, 11],
    #     "test_tgt_txt": "target_test.txt",
    #     "test_tgt_multi_view": True,
    #     "test_tgt_multi_view_index": [1, 2, 5, 11],
    # })
    
    # # 添加 tgt_all_12_views 任务
    # tasks.append({
    #     "name": "tgt_all_12_views",
    #     "tgt_domain": "Model_net_40_target_train.txt",
    #     "tgt_multi_view": True,
    #     "tgt_multi_view_index": None,
    #     "test_tgt_txt": "Model_net_40_target_test.txt",
    #     "test_tgt_multi_view": True,
    #     "test_tgt_multi_view_index": None,
    # })
    
    # # 添加 view1 到 view12 的任务
    # for i in range(1, 13):  # 1-12
    #     tasks.append({
    #         "name": f"tgt_view{i}",
    #         "tgt_domain": "Model_net_40_target_train.txt",  
    #         "tgt_multi_view": True,
    #         "tgt_multi_view_index": [i-1],
    #         "test_tgt_txt": "Model_net_40_target_test.txt",
    #         "test_tgt_multi_view": True,
    #         "test_tgt_multi_view_index": [i-1],
    #     })
    
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


def format_index_arg(indices):
	if indices is None:
		return None
	return ",".join(str(i) for i in indices)


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





def run_task(task, task_id, task_total, gpu_id, config_path, config, num_class, num_workers, test_src_txt, output_log_root):
	model_name = config["model_name"]
	datasets = config["datasets"]
	data_dir = config["data_dir"]
	src_domain = config["src_domain"]
	tgt_domain = task["tgt_domain"]		# 注意这里最好来自task

	log_info = make_log_info(task["name"])
	log_dir = os.path.join(
		"log",
		model_name,
		datasets,
		f"{src_domain}2{tgt_domain}",
		log_info,
	)
	output_dir = os.path.join(output_log_root, log_info)
	os.makedirs(output_dir, exist_ok=True)
	run_log_path = os.path.join(output_dir, "auto_run.log")
	logger = build_logger(f"auto_run.{log_info}", run_log_path)
	logger.info(
		"Start task %d/%d on GPU %d | name=%s | tgt_domain=%s | test_tgt_txt=%s",
		task_id,
		task_total,
		gpu_id,
		task["name"],
		task["tgt_domain"],
		task["test_tgt_txt"],
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
		"--tgt_domain",
		tgt_domain,
		"--tgt_multi_view",
		str(task["tgt_multi_view"]).lower(),
		"--mv_select_mode",
		task["mv_select_mode"],
		"--top_k",
		str(task["top_k"])
	]

	index_arg = format_index_arg(task["tgt_multi_view_index"])
	if index_arg is not None:
		train_cmd += ["--tgt_multi_view_index", index_arg]

	train_log = os.path.join(output_dir, "train.log")
	logger.info("Train command -> %s", train_log)
	run_command(train_cmd, REPO_ROOT, train_log, logger)

	model_path = os.path.join(log_dir, f"{model_name}.pt")
	test_tgt_txt = os.path.join(data_dir, task["test_tgt_txt"])

	source_output = os.path.join(log_dir, "source_fea.pth")
	target_output = os.path.join(log_dir, "target_fea.pth")

	extract_source_cmd = [
		"python",
		"-m",
		"retrieval.extract_fea",
		"--txt_path",
		test_src_txt,
		"--model_path",
		model_path,
		"--tgt_multi_view",
		"false",
		"--output_path",
		source_output,
		"--num_class",
		str(num_class),
		"--model_name",
		model_name,
		"--num_workers",
		str(num_workers),
		"--gpu_id",
		str(gpu_id),
		"--mv_select_mode",
		task["mv_select_mode"],
		"--top_k",
		str(task["top_k"]),
	]
	extract_source_log = os.path.join(output_dir, "extract_source.log")
	logger.info("Extract source -> %s", extract_source_log)
	run_command(extract_source_cmd, REPO_ROOT, extract_source_log, logger)

	extract_target_cmd = [
		"python",
		"-m",
		"retrieval.extract_fea",
		"--txt_path",
		test_tgt_txt,
		"--model_path",
		model_path,
		"--tgt_multi_view",
		str(task["test_tgt_multi_view"]).lower(),
		"--output_path",
		target_output,
		"--num_class",
		str(num_class),
		"--model_name",
		model_name,
		"--num_workers",
		str(num_workers),
		"--gpu_id",
		str(gpu_id),
		"--mv_select_mode",
		task["mv_select_mode"],
		"--top_k",
		str(task["top_k"]),
	]

	index_arg = format_index_arg(task["test_tgt_multi_view_index"])
	if index_arg is not None:
		extract_target_cmd += ["--tgt_multi_view_index", index_arg]
	extract_target_log = os.path.join(output_dir, "extract_target.log")
	logger.info("Extract target -> %s", extract_target_log)
	run_command(extract_target_cmd, REPO_ROOT, extract_target_log, logger)

	eval_cmd = [
		"python",
		"-m",
		"retrieval.evaluation",
		"--source_file",
		source_output,
		"--target_file",
		target_output,
	]
	eval_log = os.path.join(output_dir, "evaluation.log")
	logger.info("Evaluate -> %s", eval_log)
	run_command(eval_cmd, REPO_ROOT, eval_log, logger)
	logger.info(
		"Finish task %d/%d on GPU %d | name=%s",
		task_id,
		task_total,
		gpu_id,
		task["name"],
	)


def main():
	parser = argparse.ArgumentParser(description="Auto scheduler for VLP-CMR")
	parser.add_argument("--config", type=str, default="config/MI3DOR.yaml")
	parser.add_argument("--mem-threshold", type=int, default=100)
	parser.add_argument("--util-threshold", type=int, default=3)
	parser.add_argument("--poll-interval", type=int, default=60)
	parser.add_argument("--num-class", type=int, default=None)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--test-src-txt", type=str, required=True)		# 理论上来说我们测试集的源域都是一样的
	args = parser.parse_args()

	config_path = args.config
	config = parse_simple_yaml(config_path)

	if args.num_class is not None:
		num_class = args.num_class
	else:
		if config.get("datasets") == "MI3DOR":
			num_class = 21
		else:
			raise ValueError("num_class is required for this dataset")
	
	output_log_root = os.path.join(REPO_ROOT, "output_log", config.get("datasets"))
	os.makedirs(output_log_root, exist_ok=True)
	tasks = build_task_list()
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
			if len(free_gpus) <= 2:
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
					args=(task, task_id, task_total, gpu_id, config_path, config, num_class, args.num_workers, args.test_src_txt, output_log_root),
					daemon=True,
				)
				running[gpu_id] = worker
				worker.start()

		time.sleep(args.poll_interval)


if __name__ == "__main__":
	main()
