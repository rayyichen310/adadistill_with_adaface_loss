import logging
import os
from typing import Dict

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms

from CVLface.cvlface.research.recognition.code.run_v1.evaluations.tinyface.evaluate import evaluate


class TinyFaceDataset(torch.utils.data.Dataset):
    """Wrapper for TinyFace Arrow dataset."""

    def __init__(self, path: str):
        self.ds = load_from_disk(path)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        record = self.ds[int(idx)]
        img = record["image"].convert("RGB")
        return self.transform(img), int(record["index"])


@torch.no_grad()
def extract_tinyface_features(
    backbone: torch.nn.Module,
    dataset_path: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    flip: bool = True,
) -> np.ndarray:
    dataset = TinyFaceDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    meta = torch.load(os.path.join(dataset_path, "metadata.pt"), map_location="cpu")
    num_images = len(meta["image_paths"])

    features = None
    filled = np.zeros(num_images, dtype=bool)

    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        feats = backbone(images)
        if flip:
            images_flip = torch.flip(images, dims=[3])
            feats_flip = backbone(images_flip)
            feats = (feats + feats_flip) / 2.0
        feats = feats.detach().cpu().numpy()
        if features is None:
            feat_dim = feats.shape[1]
            features = np.zeros((num_images, feat_dim), dtype=np.float32)
        for i, index in enumerate(idx.numpy()):
            if index >= num_images:
                continue
            features[index] = feats[i]
            filled[index] = True

    if features is None:
        raise RuntimeError(f"No features extracted for tinyface dataset at {dataset_path}")
    if not filled.all():
        missing = (~filled).sum()
        logging.warning(f"TinyFace dataset {dataset_path} missing {missing} features; they remain zeros.")
    return features


def run_tinyface_evaluation(
    backbone: torch.nn.Module,
    dataset_root: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    flip: bool = True,
) -> Dict[str, float]:
    dataset_path = os.path.join(dataset_root, dataset_name)
    meta_path = os.path.join(dataset_path, "metadata.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.pt not found in {dataset_path}")

    meta = torch.load(meta_path, map_location="cpu")
    embeddings = extract_tinyface_features(
        backbone=backbone,
        dataset_path=dataset_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        flip=flip,
    )
    result = evaluate(
        all_features=embeddings,
        image_paths=meta["image_paths"],
        meta=meta,
    )
    return result
