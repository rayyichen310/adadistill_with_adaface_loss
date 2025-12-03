#!/usr/bin/env python3
"""
Standalone evaluation script for AdaDistill checkpoints.

The script loads a trained backbone from output/AdaDistill/ and runs:
 - LFW / CFP / AgeDB / CALFW / CPLFW / VGG2-FP verification (.bin files)
 - IJB-B / IJB-C evaluations (Arrow datasets)
 - TinyFace evaluation

Usage example:
    python AdaDistill/eval/run_full_eval.py \
        --checkpoint output/AdaDistill/MFN_AdaArcDistill_backbone.pth
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import torch

# Make sure "backbones", "eval", ... are importable when running the script directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backbones.iresnet import iresnet100, iresnet50, iresnet18  # noqa: E402
from backbones.mobilefacenet import MobileFaceNet  # noqa: E402
from eval import verification  # noqa: E402
from eval.ijb_evaluator import run_ijb_evaluation  # noqa: E402
from eval.tinyface_evaluator import run_tinyface_evaluation  # noqa: E402


def resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def load_config(config_path: str):
    """Load config/config.py (or an alternative path) without relying on PYTHONPATH."""
    spec = importlib.util.spec_from_file_location("adadistill_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "config"):
        raise AttributeError(f"{config_path} does not define `config`")
    return module.config


def build_backbone(network: str, embedding_size: int, use_se: bool = False) -> torch.nn.Module:
    network = network.lower()
    if network == "mobilefacenet":
        return MobileFaceNet(input_size=(112, 112), embedding_size=embedding_size)
    if network == "iresnet100":
        return iresnet100(num_features=embedding_size, use_se=use_se)
    if network == "iresnet50":
        return iresnet50(num_features=embedding_size, use_se=use_se)
    if network == "iresnet18":
        return iresnet18(num_features=embedding_size, use_se=use_se)
    raise ValueError(f"Unsupported network backbone: {network}")


def load_backbone_weights(backbone: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    logging.info("Loading checkpoint from %s", checkpoint)
    state_dict = torch.load(checkpoint, map_location=device)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys while loading checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys while loading checkpoint: %s", unexpected)


def evaluate_verification_sets(
    backbone: torch.nn.Module,
    device: torch.device,
    data_dir: str,
    targets: List[str],
    batch_size: int,
    image_size: int = 112,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if not targets:
        logging.info("No verification targets provided, skipping .bin evaluation.")
        return results

    backbone.eval()

    def forward_fn(batch: torch.Tensor) -> torch.Tensor:
        return backbone(batch.to(device, non_blocking=True))

    for name in targets:
        bin_path = os.path.join(data_dir, f"{name}.bin")
        if not os.path.exists(bin_path):
            logging.warning("Verification set %s.bin not found under %s, skipping.", name, data_dir)
            continue
        logging.info("Running verification on %s ...", name)
        data_set = verification.load_bin(bin_path, (image_size, image_size))
        with torch.no_grad():
            _, _, acc_flip, std_flip, xnorm, _ = verification.test(
                data_set=data_set,
                backbone=forward_fn,
                batch_size=batch_size,
                nfolds=10,
            )
        results[name] = {
            "Accuracy-Flip": float(acc_flip),
            "Accuracy-Std": float(std_flip),
            "XNorm": float(xnorm),
        }
        logging.info("[%s] Accuracy-Flip: %.5f +/- %.5f  XNorm: %.5f", name, acc_flip, std_flip, xnorm)
    return results


def evaluate_ijb(
    backbone: torch.nn.Module,
    device: torch.device,
    dataset_root: Optional[str],
    targets: Optional[List[str]],
    batch_size: int,
    num_workers: int,
    flip: bool,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if not dataset_root or not targets:
        logging.info("IJB evaluation skipped (missing root or targets).")
        return results
    for name in targets:
        try:
            result = run_ijb_evaluation(
                backbone=backbone,
                dataset_root=dataset_root,
                dataset_name=name,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                flip=flip,
            )
            logging.info("[IJB][%s] %s", name, json.dumps(result, indent=2))
            results[name] = {k: float(v) for k, v in result.items()}
        except FileNotFoundError as exc:
            logging.warning("IJB dataset for %s not found: %s", name, exc)
    return results


def evaluate_tinyface(
    backbone: torch.nn.Module,
    device: torch.device,
    dataset_root: Optional[str],
    targets: Optional[List[str]],
    batch_size: int,
    num_workers: int,
    flip: bool,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if not dataset_root or not targets:
        logging.info("TinyFace evaluation skipped (missing root or targets).")
        return results
    for name in targets:
        try:
            metrics = run_tinyface_evaluation(
                backbone=backbone,
                dataset_root=dataset_root,
                dataset_name=name,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                flip=flip,
            )
            logging.info("[TinyFace][%s] %s", name, json.dumps(metrics, indent=2))
            results[name] = {k: float(v) for k, v in metrics.items()}
        except FileNotFoundError as exc:
            logging.warning("TinyFace dataset for %s not found: %s", name, exc)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AdaDistill evaluation on a saved backbone.")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--config", default="config/config.py", help="Path to config file (default: config/config.py)")
    parser.add_argument(
        "--checkpoint",
        default="output/AdaDistill/MFN_AdaArcDistill_backbone.pth",
        help="Backbone checkpoint to evaluate.",
    )
    parser.add_argument("--device", default=default_device, help="Device to run evaluation on (e.g. cuda:0 or cpu).")
    parser.add_argument("--val-data", default=None, help="Directory containing *.bin verification datasets.")
    parser.add_argument("--val-targets", nargs="+", default=None, help="Override verification target list.")
    parser.add_argument("--val-batch-size", type=int, default=64, help="Batch size for pair verification datasets.")

    parser.add_argument("--ijb-root", default=None, help="Root folder with IJBB/IJBC Arrow datasets.")
    parser.add_argument("--ijb-targets", nargs="+", default=None, help="Override IJBB/IJBC target list.")
    parser.add_argument("--ijb-batch-size", type=int, default=384, help="Batch size for IJB evaluation.")
    parser.add_argument("--ijb-num-workers", type=int, default=4, help="Number of workers for IJB dataloaders.")
    parser.add_argument("--ijb-flip", action="store_true", default=None, help="Force-enable flip for IJB eval.")
    parser.add_argument(
        "--no-ijb-flip",
        dest="ijb_flip",
        action="store_false",
        help="Force-disable flip augmentation for IJB eval.",
    )

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

    parser.add_argument("--results-json", default=None, help="Optional path to dump aggregated metrics as JSON.")
    return parser.parse_args()


def run_full_evaluation(
    config: str,
    checkpoint: str,
    device: str,
    val_data: Optional[str],
    val_targets: Optional[List[str]],
    val_batch_size: int,
    ijb_root: Optional[str],
    ijb_targets: Optional[List[str]],
    ijb_batch_size: int,
    ijb_num_workers: int,
    ijb_flip: Optional[bool],
    tinyface_root: Optional[str],
    tinyface_targets: Optional[List[str]],
    tinyface_batch_size: int,
    tinyface_num_workers: int,
    tinyface_flip: Optional[bool],
    results_json: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    cfg = load_config(resolve_path(config))

    device_obj = torch.device(device)
    backbone = build_backbone(
        network=getattr(cfg, "network", "mobilefacenet"),
        embedding_size=getattr(cfg, "embedding_size", 512),
        use_se=getattr(cfg, "SE", False),
    ).to(device_obj)

    checkpoint_path = resolve_path(checkpoint)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    load_backbone_weights(backbone, checkpoint_path, device_obj)
    backbone.eval()

    val_dir = resolve_path(val_data or getattr(cfg, "val_rec", getattr(cfg, "rec", None)))
    resolved_val_targets = val_targets or list(getattr(cfg, "val_targets", []))

    ijb_root_path = resolve_path(ijb_root or getattr(cfg, "ijb_root", None))
    resolved_ijb_targets = ijb_targets or list(getattr(cfg, "ijb_targets", []))
    ijb_flip_flag = ijb_flip
    if ijb_flip_flag is None:
        ijb_flip_flag = bool(getattr(cfg, "ijb_flip", True))

    tinyface_root_path = resolve_path(tinyface_root or getattr(cfg, "tinyface_root", None))
    resolved_tinyface_targets = tinyface_targets or list(getattr(cfg, "tinyface_targets", []))
    tinyface_flip_flag = tinyface_flip
    if tinyface_flip_flag is None:
        tinyface_flip_flag = bool(getattr(cfg, "tinyface_flip", True))

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    if val_dir and os.path.isdir(val_dir):
        all_results["verification"] = evaluate_verification_sets(
            backbone=backbone,
            device=device_obj,
            data_dir=val_dir,
            targets=resolved_val_targets,
            batch_size=val_batch_size,
        )
    else:
        logging.warning("Verification data directory missing or invalid: %s", val_dir)

    all_results["ijb"] = evaluate_ijb(
        backbone=backbone,
        device=device_obj,
        dataset_root=ijb_root_path,
        targets=resolved_ijb_targets,
        batch_size=ijb_batch_size,
        num_workers=ijb_num_workers,
        flip=ijb_flip_flag,
    )

    all_results["tinyface"] = evaluate_tinyface(
        backbone=backbone,
        device=device_obj,
        dataset_root=tinyface_root_path,
        targets=resolved_tinyface_targets,
        batch_size=tinyface_batch_size,
        num_workers=tinyface_num_workers,
        flip=tinyface_flip_flag,
    )

    logging.info("Evaluation complete.")
    logging.info(json.dumps(all_results, indent=2))

    if results_json:
        results_path = resolve_path(results_json)
        with open(results_path, "w", encoding="utf-8") as fp:
            json.dump(all_results, fp, indent=2)
        logging.info("Saved metrics to %s", results_path)

    return all_results


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_full_evaluation(**vars(args))


if __name__ == "__main__":
    main()
