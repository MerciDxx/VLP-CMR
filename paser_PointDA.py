#!/usr/bin/env python3
"""Parse train.log files under specified directory and print transfer results grouped by domain transfer task."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

# 匹配 Transfer result: 数值 格式
TRANSFER_RESULT_RE = re.compile(r"Transfer result:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")

# 固定6大类输出顺序，保证表格顺序统一
GROUP_ORDER = [
    "modelnet2shapenet",
    "modelnet2scannet",
    "shapenet2modelnet",
    "shapenet2scannet",
    "scannet2modelnet",
    "scannet2shapenet",
]


def parse_last_transfer_result(path: str) -> float | None:
    """从train.log中提取最后一行Transfer result的数值"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        # 从后往前找包含 "Transfer result:" 的行
        for line in reversed(lines):
            match = TRANSFER_RESULT_RE.search(line.strip())
            if match:
                return float(match.group(1))
        return None
    except (FileNotFoundError, IOError, ValueError):
        return None


def scan_logs(root: str) -> List[Tuple[str, float | None]]:
    """递归扫描目录下所有train.log文件，提取最后一行Transfer result"""
    results: List[Tuple[str, float | None]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "train.log" not in filenames:
            continue
        log_path = os.path.join(dirpath, "train.log")
        result = parse_last_transfer_result(log_path)
        results.append((dirpath, result))
    return results


def format_folder(path: str, root: str) -> str:
    """格式化文件夹名称：只保留相对根目录路径，去除末尾时间戳后缀"""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if parts:
        parts[-1] = TIMESTAMP_RE.sub("", parts[-1])
    return "/".join(parts)


def group_items(items: List[Tuple[str, float | None]], root: str) -> Dict[str, List[Tuple[str, float | None]]]:
    """按一级分类文件夹分组，key=分类名，value=该组下所有实验"""
    groups: Dict[str, List[Tuple[str, float | None]]] = {g: [] for g in GROUP_ORDER}
    for full_path, score in items:
        rel_path = os.path.relpath(full_path, root)
        first_dir = rel_path.split(os.sep)[0]
        if first_dir in groups:
            groups[first_dir].append((full_path, score))
    # 每组内部按指标降序，None放最后
    for g in groups:
        groups[g].sort(
            key=lambda x: x[1] if x[1] is not None else float("-inf"),
            reverse=True
        )
    return groups


def print_grouped_table(items: List[Tuple[str, float | None]], root: str) -> None:
    """终端打印分组表格，每组先输出标题"""
    groups = group_items(items, root)
    headers = ["folder", "Transfer result"]
    all_rows = []
    # 先收集所有行用于计算列宽
    for g_name in GROUP_ORDER:
        exp_list = groups[g_name]
        if not exp_list:
            continue
        all_rows.append([f"===== {g_name} =====", ""])
        for path, res in exp_list:
            row = [
                format_folder(path, root),
                f"{res:.4f}" if res is not None else "N/A"
            ]
            all_rows.append(row)
    # 计算列宽度
    widths = [len(h) for h in headers]
    for row in all_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in all_rows:
        print(fmt_row(row))


def print_grouped_markdown(items: List[Tuple[str, float | None]], output_path: str | None, root: str) -> None:
    """输出带分组标题的Markdown表格"""
    groups = group_items(items, root)
    headers = ["folder", "Transfer result"]
    lines = []
    for g_name in GROUP_ORDER:
        exp_list = groups[g_name]
        if not exp_list:
            continue
        lines.append(f"### {g_name}")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for path, res in exp_list:
            row = [
                format_folder(path, root),
                f"{res:.4f}" if res is not None else "N/A"
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")  # 空行分隔不同大类
    content = "\n".join(lines) + "\n"
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        print(f"分组Markdown已保存至：{output_path}")
    else:
        print(content, end="")


def main() -> None:
    dataset_name = "PointDA"

    parser = argparse.ArgumentParser(description="Parse train.log and group results by transfer task")
    parser.add_argument(
        "root",
        nargs="?",
        default=f"output_log/{dataset_name}",
        help="Root directory to scan (default: output_log/PointDA)",
    )
    # 默认开启排序，--no-sort关闭排序
    parser.add_argument(
        "--no-sort",
        dest="sort_by",
        action="store_false",
        default=True,
        help="Disable sorting each group by transfer result descending",
    )
    parser.add_argument(
        "--md",
        nargs="?",
        default=f"{dataset_name}.md",
        help="Export grouped markdown table, e.g. --md result.md",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"目录不存在：{root}")

    items = scan_logs(root)
    if not items:
        print(f"未找到任何train.log文件：{root}")
        return

    if args.md is not None:
        print_grouped_markdown(items, args.md, root)
    else:
        print_grouped_table(items, root)


if __name__ == "__main__":
    main()