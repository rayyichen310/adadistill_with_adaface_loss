# AdaDistill with AdaFace Loss

This repository contains a PyTorch implementation of **AdaDistill** for deep face recognition, extended with:

- **AdaFace-based adaptive distillation loss** (including geometry-aware margin).
- Support for **HuggingFace CVLFace teacher models** (e.g. `cvlface_adaface_ir50_webface4m`).
- End-to-end evaluation on **LFW / CFP / AgeDB / CALFW / CPLFW / VGG2-FP**, **IJB-B/IJB-C**, and **TinyFace**.
- Multi-GPU training via `torchrun` and convenient evaluation utilities.

> 簡中/繁中使用者：本文主要以英文說明，實作與腳本名稱與原本版本相同，直接依照指令執行即可。若需要純中文版 README，可以再另外整理一份 `README_zh.md`。

---

## 1. Repository Structure

Key directories and files:

- `backbones/` – Backbone networks (e.g. `iresnet50`, `iresnet100`, `mobilefacenet`).
- `config/config.py` – Central configuration for training and evaluation (dataset, loss, teacher, LR schedule, etc.).
- `dataset/` – Dataset loaders and utilities (MXNet `.rec` loader, folder-based dataset).
- `eval/`
  - `run_full_eval.py` – Run full evaluation on a single checkpoint.
  - `run_batch_eval.py` – Run evaluation on all checkpoints in a directory and save CSV reports.
  - `verification.py`, `ijb_evaluator.py`, `tinyface_evaluator.py` – Evaluation helpers.
- `train/`
  - `train_AdaDistill.py` – Main AdaDistill training script with AdaFace loss & KD.
  - Other training variants (`train_AMLDistill.py`, `train.py`, etc.).
- `utils/`
  - `losses.py` – ArcFace / CosFace / AdaFace and adaptive KD variants.
  - `dataset.py` – Custom dataloaders with CUDA prefetch.
  - `utils_callbacks.py`, `rand_augment.py`, etc.
- `run_AdaDistill.sh` – Example launcher for `train/train_AdaDistill.py`.
- `FIXES_SUMMARY.md`, `TRAINING_SPEED_FIX.md` – Notes about bug fixes and training speed optimizations.

---

## 2. Installation

1. Create a Conda environment (Python 3.10 recommended):

   ```bash
   conda create -n adadistill python=3.10
   conda activate adadistill
   ```

2. Install PyTorch + CUDA following the official PyTorch instructions for your system:

   ```bash
   # Example (adjust CUDA version as needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install project dependencies:

   ```bash
   pip install -r requirements/requirement.txt
   ```

4. (Optional) If you want to download teacher models from HuggingFace (especially for private/protected models), set your token:

   ```bash
   export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
   ```

---

## 3. Data Preparation

### 3.1 Training Dataset (MS1MV2 / faces_emore)

The default configuration assumes the MS1MV2 dataset in InsightFace `rec` format:

- Download the MS1MV2 dataset (112×112, `train.rec` + `train.idx`).
- Place it under:

  ```text
  dataset/faces_emore/train.rec
  dataset/faces_emore/train.idx
  ```

In `config/config.py` (default):

```python
config.dataset = "emoreIresNet"
config.rec = "./dataset/faces_emore"
config.val_rec = "./dataset/faces_emore"
config.db_file_format = "rec"
config.num_classes = 85742
config.num_image = 5822653
```

Alternative dataset options (folder-based) are supported via `FaceDatasetFolder` in `utils/dataset.py` (see the `Idifface` branch in `config/config.py` for an example).

### 3.2 Verification Datasets (.bin)

Place the `.bin` files under:

```text
dataset/faces_emore/*.bin
```

The default list of verification targets is defined in `config/config.py`:

```python
config.val_targets = [
    "lfw",
    "cfp_fp",
    "cfp_ff",
    "agedb_30",
    "calfw",
    "cplfw",
    "vgg2_fp",
]
```

### 3.3 IJB-B/C and TinyFace (Arrow)

Expected directory structure:

```text
dataset/facerec_val/IJBB_gt_aligned/...
dataset/facerec_val/IJBC_gt_aligned/...
dataset/facerec_val/tinyface_aligned_pad_0.1/...
```

Relevant flags in `config/config.py`:

```python
config.eval_ijb = True
config.ijb_root = "./dataset/facerec_val"
config.ijb_targets = ["IJBB_gt_aligned", "IJBC_gt_aligned"]

config.eval_tinyface = True
config.tinyface_root = "./dataset/facerec_val"
config.tinyface_targets = ["tinyface_aligned_pad_0.1"]
```

If you do not have IJB/TinyFace prepared, you can disable them by setting `config.eval_ijb = False` and/or `config.eval_tinyface = False`.

---

## 4. Configuration (`config/config.py`)

All training and evaluation options are centralized in `config/config.py`. A simplified overview of the most important fields:

```python
from easydict import EasyDict as edict

config = edict()

# Dataset & output
config.dataset = "emoreIresNet"          # emoreIresNet / Idifface / CASIA_WebFace
config.output = "output/AdaDistill_sync/"

# Embedding & loss
config.embedding_size = 512
config.loss = "AdaFace"                  # ArcFace / CosFace / AdaFace / MLLoss
config.s = 64.0
config.m = 0.4
config.h = 0.333
config.t_alpha = 0.01
config.adaptive_alpha = True

# Geometry-aware KD margin (AdaFace variant)
config.use_geom_margin = True
config.geom_margin_w = 1.0
config.geom_margin_k = 3.0
config.geom_margin_warmup_epoch = 1

# Student network
config.network = "mobilefacenet"         # iresnet100 / iresnet50 / iresnet18 / mobilefacenet
config.teacher = "cvlface_ir50"
config.SE = False

# Teacher weights / HuggingFace
config.pretrained_teacher_path = "output/teacher/295672backbone.pth"
config.pretrained_teacher_header_path = "output/teacher/295672header.pth"
config.teacher_repo_id = "minchul/cvlface_adaface_ir50_webface4m"
config.teacher_cache = "~/.cvlface_cache/minchul/cvlface_adaface_ir50_webface4m"

# Optimization
config.batch_size = 384
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# Training state
config.global_step = 0   # 0 => from scratch; >0 => resume from checkpoint
```

To resume training from a given checkpoint (e.g. `output/AdaDistill_sync/212282backbone.pth`), set:

```python
config.output = "output/AdaDistill_sync/"
config.global_step = 212282
```

The learning rate schedule is defined by `config.lr_func` and depends on `config.dataset`. For `emoreIresNet`, the LR uses a warmup phase followed by step decay.

---

## 5. Training

### 5.1 Single-node Training with `torchrun`

From the repository root:

```bash
cd /path/to/AdaDistill
bash run_AdaDistill.sh
```

`run_AdaDistill.sh` (simplified):

```bash
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train/train_AdaDistill.py
```

To train on multiple GPUs on a single node, adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node`, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train/train_AdaDistill.py
```

Key behavior in `train/train_AdaDistill.py`:

- Uses DistributedDataParallel for multi-GPU training.
- Supports optional `config.run_eval_at_start` to run evaluation right after loading a checkpoint.
- Uses per-step LR warmup inside the first `config.warmup_epoch` epochs.
- AdaFace loss (`AdaptiveAAdaFace`) is updated with student–teacher similarity and optional geometry-aware margin.

### 5.2 Other Training Scripts

- `run_AMLDistill.sh` / `train/train_AMLDistill.py` – Alternative distillation variant.
- `run_standalone.sh` / `train/train.py` – Non-distillation / baseline training script.

---

## 6. Evaluation

### 6.1 Full Evaluation on a Single Checkpoint

Use `eval/run_full_eval.py`:

```bash
cd /path/to/AdaDistill

python eval/run_full_eval.py \
  --checkpoint output/AdaDistill/MFN_AdaArcDistill_backbone.pth \
  --config config/config.py \
  --val-data ./dataset/faces_emore \
  --ijb-root ./dataset/facerec_val \
  --tinyface-root ./dataset/facerec_val \
  --device cuda:0
```

This script:

- Builds the backbone according to `config.network` and `config.embedding_size`.
- Loads the checkpoint into the backbone.
- Runs:
  - LFW / CFP-FP / CFP-FF / AgeDB-30 / CALFW / CPLFW / VGG2-FP verification.
  - IJB-B / IJB-C evaluation (if enabled and data available).
  - TinyFace evaluation (if enabled and data available).

Results are logged as a structured JSON-like dictionary (accuracy, XNorm, TPR@FPR, rank-N, etc.).

### 6.2 Batch Evaluation over Many Checkpoints

To evaluate all checkpoints in a directory and save CSV reports, use `eval/run_batch_eval.py`:

```bash
python eval/run_batch_eval.py \
  --config config/config.py \
  --checkpoint-dir output/AdaDistill \
  --checkpoint-suffix backbone.pth \
  --device cuda:0 \
  --val-data ./dataset/faces_emore \
  --ijb-root ./dataset/facerec_val \
  --tinyface-root ./dataset/facerec_val \
  --save-json
```

This will:

- Find all `*backbone.pth` checkpoints under `output/AdaDistill`.
- Run `run_full_evaluation` for each checkpoint.
- Save flattened metrics as CSV files under `<checkpoint-dir>/reports/`.
- Optionally save the raw JSON metrics when `--save-json` is provided.

You can override `--val-targets`, `--ijb-targets`, `--tinyface-targets`, and the batch sizes via CLI arguments.

---

## 7. Known Issues and Fixes

See `FIXES_SUMMARY.md` and `TRAINING_SPEED_FIX.md` for detailed notes. Highlights:

- **Shape mismatch in AdaFace loss** – Fixed in `utils/losses.py` to avoid tensor shape mismatches when computing logits.
- **Training speed degradation** – Original evaluation callbacks were invoked at every step; they are now guarded by frequency checks (e.g. `val_eval_every_n_epoch`, `eval_every_n_epoch`) in `train/train_AdaDistill.py`.
- **Teacher loading issues** – If `pretrained_teacher_path` or `pretrained_teacher_header_path` do not exist, logs may show `teacher init, failed!`. Ensure correct paths or provide a valid HuggingFace teacher (`config.teacher_repo_id`) and cache directory.
- **NaN loss** – Can be caused by invalid teacher states or overly aggressive LR. Check teacher initialization, lower the LR, and monitor gradients (gradient clipping is already enabled).

---

## 8. Citation

If you find this code useful in your research, please cite the original AdaDistill paper:

```bibtex
@InProceedings{Boutros_2024_ECCV,
  author    = {Fadi Boutros and Vitomir {\v{S}}truc and Naser Damer},
  title     = {AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  booktitle = {Computer Vision -- ECCV 2024},
  month     = {October},
  year      = {2024}
}
```

---

## 9. License

This repository is a research-oriented re-implementation and extension of the original AdaDistill project. The original project is released under the **Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license; by using this code, you agree to respect the same non-commercial terms and attribution requirements. For details, please refer to the original AdaDistill repository and its LICENSE.
