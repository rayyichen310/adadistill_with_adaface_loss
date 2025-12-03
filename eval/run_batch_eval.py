#!/usr/bin/env python3
"""Batch evaluation utility for running AdaDistill metrics over many checkpoints."""

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from eval.run_full_eval import run_full_evaluation, resolve_path


def flatten_results(tree: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    """Flatten nested metric dictionaries into dot-separated keys."""
    flat: Dict[str, float] = {}

    def _walk(prefix: str, node):
        if isinstance(node, dict):
            for key, value in node.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                _walk(next_prefix, value)
        else:
            flat[prefix] = float(node)

    _walk("", tree)
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on every checkpoint inside a directory.")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--config", default="config/config.py", help="Path to config file (default: config/config.py)")
    parser.add_argument("--checkpoint-dir", default="output/AdaDistill", help="Directory containing *_backbone.pth files.")
    parser.add_argument("--device", default=default_device, help="Device to run evaluation on (e.g. cuda:0 or cpu).")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory to store CSV reports (default: <checkpoint-dir>/reports).",
    )
    parser.add_argument("--save-json", action="store_true", help="Also persist raw JSON metrics next to the CSV report.")

    parser.add_argument("--val-data", default=None, help="Directory containing *.bin verification datasets.")
    parser.add_argument("--val-targets", nargs="+", default=None, help="Override verification target list.")
    parser.add_argument("--val-batch-size", type=int, default=64, help="Batch size for pair verification datasets.")

    parser.add_argument("--ijb-root", default=None, help="Root folder with IJBB/IJBC Arrow datasets.")
    parser.add_argument("--ijb-targets", nargs="+", default=None, help="Override IJBB/IJBC target list.")
    parser.add_argument("--ijb-batch-size", type=int, default=384, help="Batch size for IJB evaluation.")
    parser.add_argument("--ijb-num-workers", type=int, default=4, help="Number of workers for IJB dataloaders.")
    parser.add_argument("--ijb-flip", action="store_true", default=None, help="Force-enable flip for IJB eval.")
    parser.add_argument("--no-ijb-flip", dest="ijb_flip", action="store_false", help="Force-disable flip augmentation for IJB eval.")

    parser.add_argument("--tinyface-root", default=None, help="Root folder with TinyFace Arrow dataset.")
    parser.add_argument("--tinyface-targets", nargs="+", default=None, help="Override TinyFace target list.")
    parser.add_argument("--tinyface-batch-size", type=int, default=384, help="Batch size for TinyFace evaluation.")
    parser.add_argument("--tinyface-num-workers", type=int, default=4, help="Number of workers for TinyFace dataloaders.")
    parser.add_argument("--tinyface-flip", action="store_true", default=None, help="Force-enable flip for TinyFace eval.")
    parser.add_argument(
        "--no-tinyface-flip",
        dest="tinyface_flip",
        action="store_false",
        help="Force-disable flip augmentation for TinyFace eval.",
    )
    return parser.parse_args()


def ensure_report_dir(path: Optional[str], checkpoint_dir: str) -> str:
    report_dir = path or os.path.join(checkpoint_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def collect_checkpoints(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    checkpoints = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("backbone.pth")]
    checkpoints.sort()
    if not checkpoints:
        raise RuntimeError(f"No *_backbone.pth files found under {directory}")
    return checkpoints


def write_csv_report(path: str, metrics: Dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, f"{metrics[key]:.6f}"])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    checkpoint_dir = resolve_path(args.checkpoint_dir)
    report_dir = ensure_report_dir(resolve_path(args.report_dir) if args.report_dir else None, checkpoint_dir)
    checkpoints = collect_checkpoints(checkpoint_dir)

    logging.info("Found %d checkpoints under %s", len(checkpoints), checkpoint_dir)
    for checkpoint in checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0]
        logging.info("Evaluating %s", ckpt_name)
        json_path = os.path.join(report_dir, f"{ckpt_name}_metrics.json") if args.save_json else None
        results = run_full_evaluation(
            config=args.config,
            checkpoint=checkpoint,
            device=args.device,
            val_data=args.val_data,
            val_targets=args.val_targets,
            val_batch_size=args.val_batch_size,
            ijb_root=args.ijb_root,
            ijb_targets=args.ijb_targets,
            ijb_batch_size=args.ijb_batch_size,
            ijb_num_workers=args.ijb_num_workers,
            ijb_flip=args.ijb_flip,
            tinyface_root=args.tinyface_root,
            tinyface_targets=args.tinyface_targets,
            tinyface_batch_size=args.tinyface_batch_size,
            tinyface_num_workers=args.tinyface_num_workers,
            tinyface_flip=args.tinyface_flip,
            results_json=json_path,
        )
        flat_metrics = flatten_results(results)
        csv_path = os.path.join(report_dir, f"{ckpt_name}_metrics.csv")
        write_csv_report(csv_path, flat_metrics)
        logging.info("Saved CSV report to %s", csv_path)


if __name__ == "__main__":
    main()