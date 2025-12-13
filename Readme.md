# AdaDistill: Adaptive Knowledge Distillation with AdaFace Loss

This repository contains a PyTorch implementation of **AdaDistill** for deep face recognition, extended with **AdaFace Loss** and **Geometry-aware Margin**. It supports training with HuggingFace teacher models, multi-GPU training via `torchrun`, and comprehensive evaluation pipelines.

## Key Features

- **Adaptive Distillation**: Implements AdaDistill with adaptive distillation loss.
- **AdaFace Integration**: Extends the original framework with AdaFace loss for better robustness.
- **Geometry-aware Margin**: Introduces a geometry-aware margin mechanism to refine the distillation process based on student-teacher angular alignment.
- **HuggingFace Teacher Support**: Seamlessly load teacher models (e.g., `cvlface_adaface_ir50_webface4m`) from HuggingFace Hub (requires `CVLface`).
- **Optimized Training**: Includes fixes for training speed degradation and shape mismatch issues (see `TRAINING_SPEED_FIX.md` and `FIXES_SUMMARY.md`).
- **Comprehensive Evaluation**: End-to-end evaluation on LFW, CFP-FP, AgeDB, CALFW, CPLFW, VGG2-FP, IJB-B/C, and TinyFace.

## Installation

1. **Create a Conda Environment** (Python 3.10 recommended):
   ```bash
   conda create -n adadistill python=3.10
   conda activate adadistill
   ```

2. **Install PyTorch**:
   Follow the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version. Example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements/requirement.txt
   ```

4. **(Optional) Install CVLface**:
   Required for loading HuggingFace teacher models and running IJB/TinyFace evaluations.
   ```bash
   # Follow instructions at https://github.com/mk-minchul/CVLface
   # Or install if available in requirements
   ```

## Data Preparation

### Directory Structure
Ensure your dataset directory looks like this:

```text
dataset/
├── faces_emore/              # Training Data (MS1MV2)
│   ├── train.rec
│   ├── train.idx
│   ├── lfw.bin
│   ├── cfp_fp.bin
│   └── ...
└── facerec_val/              # Evaluation Data (IJB, TinyFace)
    ├── IJBB_gt_aligned/
    ├── IJBC_gt_aligned/
    └── tinyface_aligned_pad_0.1/
```

### Training Data
The default configuration uses **MS1MV2** in InsightFace `.rec` format. Place `train.rec` and `train.idx` in `dataset/faces_emore/`.

### Validation Data
- **Standard Benchmarks**: Place `.bin` files (LFW, CFP-FP, AgeDB, etc.) in `dataset/faces_emore/`.
- **IJB & TinyFace**: Place aligned images in `dataset/facerec_val/`.

## Configuration

All configurations are managed in `config/config.py`. Key parameters include:

- **Dataset**: `config.dataset = "emoreIresNet"`
- **Loss**: `config.loss = "AdaFace"` (Options: ArcFace, CosFace, AdaFace)
- **Teacher**:
  - `config.teacher = "cvlface_ir50"` (requires HuggingFace token)
  - Or `config.teacher = "iresnet50"` for local models.
- **Geometry-aware Margin**:
  - `config.use_geom_margin = True`
  - `config.geom_margin_k = 3.0`

## Usage

### Training

**Single-Node Multi-GPU Training**:
Use the provided script `run_AdaDistill.sh` which uses `torchrun`:

```bash
bash run_AdaDistill.sh
```

To customize GPU usage, edit the script:
```bash
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train/train_AdaDistill.py
```

**Resume Training**:
Set `config.global_step` in `config/config.py` to the step number you want to resume from (e.g., `212282`).

### Evaluation

**Single Checkpoint Evaluation**:
```bash
python eval/run_full_eval.py \
  --checkpoint output/AdaDistill/your_model_backbone.pth \
  --config config/config.py \
  --val-data ./dataset/faces_emore \
  --device cuda:0
```

**Batch Evaluation**:
Evaluate all checkpoints in a directory and generate a CSV report:
```bash
python eval/run_batch_eval.py \
  --config config/config.py \
  --checkpoint-dir output/AdaDistill \
  --checkpoint-suffix backbone.pth \
  --save-json
```

## Troubleshooting & Recent Fixes

Please refer to the following documents for detailed fix information:

- **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)**: Details on fixing the "Shape mismatch" error in loss calculation and handling missing `CVLface` dependencies.
- **[TRAINING_SPEED_FIX.md](TRAINING_SPEED_FIX.md)**: Details on optimizing training speed by reducing the frequency of evaluation callbacks.

**Common Issues:**
- **NaN Loss**: Often caused by improper teacher initialization or high learning rates. Ensure your teacher model is loaded correctly.
- **Missing CVLface**: If you see import errors related to `CVLface`, either install the library or switch to a local teacher model in `config.py`.

## Citation

If you use this code, please cite the original AdaDistill paper:

```bibtex
@InProceedings{Boutros_2024_ECCV,
  author    = {Fadi Boutros and Vitomir {\v{S}}truc and Naser Damer},
  title     = {AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  booktitle = {Computer Vision -- ECCV 2024},
  month     = {October},
  year      = {2024}
}
```

## License

This project is released under the **Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.
