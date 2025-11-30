import logging
import os
from typing import Dict, List

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms

from CVLface.cvlface.research.recognition.code.run_v1.evaluations.ijbbc.evaluate import evaluate


class IJBDataset(torch.utils.data.Dataset):
    """Simple wrapper to read IJB arrow datasets from disk."""

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
        img = record["image"]
        img = img.convert("RGB")
        img = self.transform(img)
        return img, int(record["index"])


@torch.no_grad()
def extract_ijb_features(
    backbone: torch.nn.Module,
    dataset_path: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    flip: bool = True,
) -> np.ndarray:
    """Run backbone on IJB dataset and return features aligned by index."""
    dataset = IJBDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    meta_path = os.path.join(dataset_path, "metadata.pt")
    meta = torch.load(meta_path, map_location="cpu")
    num_images = len(meta["templates"])

    feat_dim = None
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
        if feat_dim is None:
            feat_dim = feats.shape[1]
            features = np.zeros((num_images, feat_dim), dtype=np.float32)
        for i, index in enumerate(idx.numpy()):
            if index >= num_images:
                continue
            features[index] = feats[i]
            filled[index] = True

    if features is None:
        raise RuntimeError(f"No features extracted for dataset at {dataset_path}")
    if not filled.all():
        missing = (~filled).sum()
        logging.warning(f"IJB dataset {dataset_path} has {missing} missing features; they will remain zeros.")
    return features


def run_ijb_evaluation(
    backbone: torch.nn.Module,
    dataset_root: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    flip: bool = True,
) -> Dict[str, float]:
    """Evaluate backbone on a single IJB dataset and return metrics."""
    dataset_path = os.path.join(dataset_root, dataset_name)
    meta_path = os.path.join(dataset_path, "metadata.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.pt not found in {dataset_path}")

    meta = torch.load(meta_path, map_location="cpu")
    embeddings = extract_ijb_features(
        backbone=backbone,
        dataset_path=dataset_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        flip=flip,
    )

    # Ensure shape alignment
    if embeddings.shape[0] != len(meta["faceness_scores"]):
        logging.warning(
            f"Embeddings count ({embeddings.shape[0]}) != faceness_scores ({len(meta['faceness_scores'])}); "
            "evaluation may be inconsistent."
        )

    result = evaluate(
        embeddings=embeddings,
        faceness_scores=np.asarray(meta["faceness_scores"]),
        templates=np.asarray(meta["templates"]),
        medias=np.asarray(meta["medias"]),
        label=np.asarray(meta["label"]),
        p1=np.asarray(meta["p1"]),
        p2=np.asarray(meta["p2"]),
        dummy=False,
    )
    return result
