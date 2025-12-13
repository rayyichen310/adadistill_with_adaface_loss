# AdaDistill with AdaFace and Geometry-aware Margin

A PyTorch implementation of **AdaDistill** for face recognition, extended with
**AdaFace loss** and a **geometry-aware distillation margin** for more robust
and efficient student–teacher alignment.



This repository extends **AdaDistill (ECCV 2024)** with:
- AdaFace-based quality-aware margin adaptation
- A geometry-aware distillation margin guided by teacher confidence



---

## 1. What’s New in This Repo

- **AdaFace-based distillation**  
  Applies norm-based adaptive margins to reduce over-penalization of
  low-quality samples during knowledge distillation.

- **Geometry-aware margin**  
  Adds an extra margin when the teacher is confident but the student is
  geometrically misaligned.

- **HuggingFace teacher support**  
  Seamlessly loads CVLFace pretrained models from HuggingFace Hub.

- **Comprehensive evaluation**  
  Supports LFW, CFP-FP, AgeDB, CALFW, CPLFW, VGG2-FP, IJB-B/C, and TinyFace.




---

## 2. Key Contributions

### 2.1 AdaFace-based Adaptive Margin

AdaFace uses the **L2 norm of the embedding** as a proxy for image quality.

- High-norm embeddings (high-quality samples) receive larger angular margins.
- Low-norm embeddings (low-quality or ambiguous samples) receive smaller margins.

This design prevents the student from overfitting to noisy or low-quality samples
during distillation and encourages more robust feature learning.

---

### 2.2 Geometry-aware Distillation Margin

In addition to norm-based adaptation, this repository introduces a
**geometry-aware margin** for knowledge distillation.

For each training sample, the following similarities are considered:

- Student–Teacher similarity:  
  `cos(f_s, f_t)`
- Teacher confidence for the ground-truth class:  
  `cos(w_y, f_t)`

When the teacher exhibits high class confidence but the student remains
geometrically misaligned, an additional margin penalty is applied.

This mechanism:

- Emphasizes informative and learnable samples
- Encourages the student feature space to match the teacher’s geometry
- Avoids unnecessary penalties on already aligned samples

---



## 3. Installation

### 3.1 Environment Setup

```bash
conda create -n adadistill python=3.10
conda activate adadistill
```

Install PyTorch according to your CUDA version.

---

### 3.2 Dependencies

```bash
pip install -r requirements/requirement.txt
```

---

### 3.3 Optional: CVLFace

CVLFace is required for:

- Loading HuggingFace CVLFace teacher models
- Running IJB-B/C and TinyFace evaluations

Installation guide:  
https://github.com/mk-minchul/CVLface

---

## 4. Data Preparation

### 4.1 Training Data

- Dataset: **MS1MV2**
- Format: InsightFace `.rec` / `.idx`
- Path: `dataset/faces_emore/`

---

### 4.2 Evaluation Data

- Standard benchmarks (`.bin`):  
  LFW, CFP-FP, AgeDB, CALFW, CPLFW, VGG2-FP
- Large-scale benchmarks:  
  IJB-B, IJB-C, TinyFace (requires CVLFace)

---

## 5. Configuration

All settings are managed in `config/config.py`.

```python
config.dataset = "emoreIresNet"
config.loss = "AdaFace"

config.use_geom_margin = True
config.geom_margin_w = 1.0
config.geom_margin_k = 3.0
config.geom_margin_warmup_epoch = 1

config.teacher = "cvlface_ir50"   # or local pretrained teacher
```

---

## 6. Training

Multi-GPU training (single node):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train/train_AdaDistill.py
```

Resume training by setting `config.global_step` to the desired step.

---

## 7. Evaluation

### 7.1 Single Checkpoint Evaluation

```bash
python eval/run_full_eval.py \
  --checkpoint output/AdaDistill/your_model_backbone.pth \
  --config config/config.py
```

---

### 7.2 Batch Evaluation

```bash
python eval/run_batch_eval.py \
  --checkpoint-dir output/AdaDistill \
  --checkpoint-suffix backbone.pth \
  --save-json
```

This generates CSV (and optional JSON) summaries for all checkpoints.

---

## 8. Troubleshooting

- **FIXES_SUMMARY.md**  
  Shape mismatch fixes and CVLFace dependency handling.

- **TRAINING_SPEED_FIX.md**  
  Training speed optimization by reducing evaluation frequency.

---

## 9. Citation

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

---

## 10. License

This project is released under the  
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.

- Non-commercial use only  
- Attribution required  
- Derivative works must use the same license
