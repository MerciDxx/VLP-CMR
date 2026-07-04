#!/usr/bin/env python3
"""Parse train.log files under specified directory and print transfer results."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

# 匹配 Transfer result: 数值 格式
TRANSFER_RESULT_RE = re.compile(r"Transfer result:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


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
    """扫描目录下所有train.log文件，提取最后一行Transfer result"""
    results: List[Tuple[str, float | None]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "train.log" not in filenames:
            continue
        log_path = os.path.join(dirpath, "train.log")
        result = parse_last_transfer_result(log_path)
        results.append((dirpath, result))
    return results


def format_folder(path: str, root: str) -> str:
    """格式化文件夹名称，去除时间戳后缀"""
    base_root = root
    if os.path.basename(root) != "output_log":
        base_root = os.path.dirname(root)
    rel = os.path.relpath(path, base_root) if path.startswith(base_root) else os.path.basename(path)
    parts = rel.split(os.sep)
    if parts:
        parts[-1] = TIMESTAMP_RE.sub("", parts[-1])
    return "/".join(parts)


def print_table(items: List[Tuple[str, float | None]], root: str) -> None:
    """打印表格到终端"""
    headers = ["folder", "Transfer result"]
    
    rows: List[List[str]] = []
    for path, result in items:
        row = [
            format_folder(path, root),
            f"{result:.4f}" if result is not None else "N/A"
        ]
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def print_markdown_table(
    items: List[Tuple[str, float | None]],
    output_path: str | None,
    root: str,
) -> None:
    """输出Markdown格式表格"""
    headers = ["folder", "Transfer result"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for path, result in items:
        row = [
            format_folder(path, root),
            f"{result:.4f}" if result is not None else "N/A"
        ]
        lines.append("| " + " | ".join(row) + " |")

    content = "\n".join(lines) + "\n"
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
    else:
        print(content, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse train.log files and extract transfer results")
    parser.add_argument(
        "root",
        nargs="?",
        default="output_log/PointDA",
        help="Root directory to scan (default: output_log/PointDA)",
    )
    parser.add_argument(
        "--sort",
        dest="sort_by",
        action="store_false",
        help="Sort by transfer result (descending)",
    )
    parser.add_argument(
        "--md",
        dest="md_output",
        nargs="?",
        const="results.md",
        default="results.md",
        help="Write Markdown table to file (optional: specify filename, default: results.md)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")

    items = scan_logs(root)
    if not items:
        print(f"No train.log files found under {root}")
        return

    # 排序（按结果降序，None值排在最后）
    if args.sort_by:
        items.sort(
            key=lambda item: item[1] if item[1] is not None else float("-inf"),
            reverse=True
        )

    if args.md_output is not None:
        # 如果提供了 --md 但没有指定文件名，使用默认值
        output_path = args.md_output if args.md_output != "results.md" else args.md_output
        print_markdown_table(items, output_path, root)
    else:
        print_table(items, root)


if __name__ == "__main__":
    main()