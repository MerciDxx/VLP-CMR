#!/usr/bin/env python3
"""Parse evaluation.log files under output_log and print metrics."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

METRIC_RE = re.compile(r"^([A-Za-z0-9_\-]+)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")


def parse_metrics(path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = METRIC_RE.match(line.strip())
            if not match:
                continue
            key, value = match.group(1), match.group(2)
            metrics[key] = float(value)
    return metrics


def scan_logs(root: str) -> List[Tuple[str, Dict[str, float]]]:
    results: List[Tuple[str, Dict[str, float]]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "evaluation.log" not in filenames:
            continue
        log_path = os.path.join(dirpath, "evaluation.log")
        metrics = parse_metrics(log_path)
        results.append((dirpath, metrics))
    return results


def collect_keys(items: List[Tuple[str, Dict[str, float]]]) -> List[str]:
    preferred = ["NN", "FT", "ST", "F-Measure", "DCG", "ANMRR", "AUC_0"]
    keys = set()
    for _path, metrics in items:
        keys.update(metrics.keys())
    ordered = [key for key in preferred if key in keys]
    remaining = sorted(key for key in keys if key not in preferred)
    return ordered + remaining


def format_folder(path: str, root: str) -> str:
    base_root = root
    if os.path.basename(root) != "output_log":
        base_root = os.path.dirname(root)
    rel = os.path.relpath(path, base_root) if path.startswith(base_root) else os.path.basename(path)
    parts = rel.split(os.sep)
    if parts:
        parts[-1] = TIMESTAMP_RE.sub("", parts[-1])
    return "/".join(parts)


def print_table(items: List[Tuple[str, Dict[str, float]]], keys: List[str], root: str) -> None:
    headers = ["folder"] + keys
    rows: List[List[str]] = []
    for path, metrics in items:
        row = [format_folder(path, root)]
        for key in keys:
            value = metrics.get(key)
            row.append("" if value is None else f"{value:.4f}")
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
    items: List[Tuple[str, Dict[str, float]]],
    keys: List[str],
    output_path: str | None,
    root: str,
) -> None:
    headers = ["folder"] + keys
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for path, metrics in items:
        row = [format_folder(path, root)]
        for key in keys:
            value = metrics.get(key)
            row.append("" if value is None else f"{value:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    content = "\n".join(lines) + "\n"
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(content)
    else:
        print(content, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse evaluation.log metrics under output_log")
    parser.add_argument(
        "root",
        nargs="?",
        default="output_log/MI3DOR",
        help="Root directory to scan (default: output_log)",
    )
    parser.add_argument(
        "--sort-by",
        dest="sort_by",
        default='NN',
        help="Metric name to sort by (descending).",
    )
    parser.add_argument(
        "--md",
        dest="md_output",
        default="MI3DOR.md",
        help="Write Markdown table to file (or '-' for stdout).",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")

    items = scan_logs(root)
    if not items:
        print(f"No evaluation.log files found under {root}")
        return

    keys = collect_keys(items)
    if args.sort_by:
        if args.sort_by not in keys:
            available = ", ".join(keys)
            raise SystemExit(f"Unknown metric: {args.sort_by}. Available: {available}")
        items.sort(key=lambda item: item[1].get(args.sort_by, float("-inf")), reverse=True)
    if args.md_output:
        output_path = None if args.md_output == "-" else args.md_output
        print_markdown_table(items, keys, output_path, root)
    else:
        print_table(items, keys, root)


if __name__ == "__main__":
    main()
