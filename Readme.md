# AdaDistill with AdaFace and Geometry-aware Margin

A PyTorch implementation of **AdaDistill** for face recognition, extended with
**AdaFace loss** and a **geometry-aware distillation margin** for more robust
and efficient student–teacher alignment.

---

## Quick Overview

This repository extends **AdaDistill (ECCV 2024)** with:

- AdaFace-based adaptive margins using feature norm as a quality proxy
- Geometry-aware distillation margin for improved student–teacher alignment
- HuggingFace CVLFace teacher model support
- End-to-end evaluation on standard and large-scale face recognition benchmarks

---

## What’s New in This Repo

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

## Installation

```bash
conda create -n adadistill python=3.10
conda activate adadistill
pip install -r requirements/requirement.txt
```

---

## Training

Multi-GPU training (single node):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train/train_AdaDistill.py
```

---

## Evaluation

Single checkpoint evaluation:

```bash
python eval/run_full_eval.py \
  --checkpoint path/to/checkpoint.pth \
  --config config/config.py
```

Batch evaluation:

```bash
python eval/run_batch_eval.py \
  --checkpoint-dir output/AdaDistill \
  --config config/config.py
```

---

## Notes

- CVLFace is required only for HuggingFace teachers and IJB/TinyFace evaluation.
- Standard `.bin` verification (LFW, CFP, AgeDB) works without CVLFace.

---

## Detailed Documentation

### AdaFace-based Adaptive Margin

AdaFace uses the L2 norm of the embedding as a proxy for image quality.
High-quality samples receive larger angular margins, while low-quality samples
receive smaller margins, improving training stability.

---

### Geometry-aware Distillation Margin

This repository introduces a geometry-aware margin that explicitly considers
student–teacher alignment.

For each sample:
- Student–Teacher similarity: `cos(f_s, f_t)`
- Teacher class confidence: `cos(w_y, f_t)`

When the teacher is confident but the student is not yet well aligned, an
additional margin penalty is applied, encouraging closer geometric alignment.

---

## Configuration

All settings are managed in `config/config.py`.

```python
config.dataset = "emoreIresNet"
config.loss = "AdaFace"

config.use_geom_margin = True
config.geom_margin_w = 1.0
config.geom_margin_k = 3.0
config.geom_margin_warmup_epoch = 1

config.teacher = "cvlface_ir50"  # or local pretrained teacher
```

---

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

---

## License

This project is released under the  
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.

- Non-commercial use only  
- Attribution required  
- Derivative works must use the same license

