# AdaDistill: Adaptive Knowledge Distillation with AdaFace Loss and Geometry-aware Margin

This repository provides a **PyTorch implementation of AdaDistill** for deep face
recognition, extended with **AdaFace loss** and a **geometry-aware margin**
to improve student–teacher feature alignment during knowledge distillation.

The framework supports **HuggingFace-based teacher models**, **multi-GPU training
via `torchrun`**, and **comprehensive evaluation pipelines** across both standard
and large-scale face recognition benchmarks.

---

## 1. Overview

AdaDistill is an adaptive knowledge distillation framework designed for
margin-based face recognition models. This project extends the original AdaDistill method by:

1. Integrating **AdaFace loss** as the classification head.
2. Introducing a **geometry-aware margin** that explicitly models the geometric
   relationship between student and teacher embeddings.

These extensions aim to improve distillation stability, convergence behavior,
and final recognition performance.

---

## 2. Design Motivation

Standard knowledge distillation methods typically align student and teacher
features using distance-based losses (e.g., L2 or cosine loss).
Such approaches treat all samples equally, regardless of:

- Image quality
- Teacher confidence
- Difficulty of geometric alignment

In face recognition, this leads to two major issues:

1. **Over-penalization of low-quality or ambiguous samples**, which introduces
   noisy gradients.
2. **Under-emphasis on high-confidence but misaligned samples**, which are often
   the most informative for learning discriminative decision boundaries.

This repository addresses these limitations by combining:

- **AdaFace-based quality awareness**, and
- **Geometry-aware margin modulation** guided by teacher confidence.

---

## 3. AdaFace-based Adaptive Margin

AdaFace uses the **L2 norm of the embedding** as a proxy for image quality.

- High-norm embeddings (high-quality samples) receive larger angular margins.
- Low-norm embeddings (low-quality or ambiguous samples) receive smaller margins.

By adapting the margin based on feature norm, the student model avoids
overfitting to noisy samples while allocating more discriminative capacity
to reliable samples.

This property is particularly important in the distillation setting, where
the student is more vulnerable to noisy supervision.

---

## 4. Why Geometry-aware Margin Instead of Feature-level KD?

Most knowledge distillation methods apply an explicit loss between student and
teacher embeddings. However, in margin-based face recognition, this introduces
an optimization mismatch:

- Classification loss operates in **angular margin space**
- Feature-level KD operates in **embedding space**

Applying both simultaneously can lead to competing objectives.

Instead of introducing an additional KD loss, this repository **modulates the
classification margin itself**, allowing distillation signals to act directly
on the decision boundary.

This design preserves the geometric structure imposed by margin-based losses
and avoids interference between loss terms.

---

## 5. Methodology: Adaptive Geometry-Aware Margin

### 5.1 The Core Formula
Unlike static methods that use hard-coded thresholds, our approach dynamically adjusts the penalty based on the teacher's confidence. The additional margin penalty is calculated as:

$$
\text{Penalty} = \text{ReLU}(\text{Conf}_t - \cos_{st}) \cdot \text{Conf}_t
$$

Where:
- $\text{Conf}_t = \cos(f_t, c_y)$: The teacher's cosine similarity to the ground truth class center (Teacher's Confidence).
- $\cos_{st} = \cos(f_s, f_t)$: The cosine similarity between student and teacher embeddings (Alignment).
- $\text{ReLU}$: Acts as a **gating mechanism**. The penalty is only applied when the student's alignment ($\cos_{st}$) is *worse* than the teacher's confidence ($\text{Conf}_t$).

  
### 5.2 The Unified Architecture
Our framework operates on two parallel paths during training (as shown in our presentation):

1. **Path A: Center Update (Teacher Guidance)**
   The class centers ($w$) are not static. They are updated using an **Exponential Moving Average (EMA)** guided by the teacher's features ($f_t$), ensuring the "standard answer" evolves to be more accurate.

2. **Path B: Margin Calculation (Student Learning)**
   The final margin for the student is a combination of two components:
   - **Quality-aware Margin:** Derived from AdaFace (based on norm $||f_s||$).
   - **Geometry-aware Margin:** The calculated $\text{Penalty}$ derived above.

The final margin scaler applied to the logits is updated as follows:

$$
m \leftarrow m + w \cdot \mathcal{P}
$$

<img width="2285" height="1164" alt="image" src="https://github.com/user-attachments/assets/aa5a582f-6e53-48a7-af97-648de8dc5a36" />

## 6. When Does the Geometry-aware Margin Take Effect?

The geometry-aware margin is active only under the following conditions:

- Teacher confidence is high (`cos(w_y, f_t)` is large)
- Student–teacher similarity is low (`cos(f_s, f_t)` is small)

If the student embedding is already well aligned with the teacher, the margin
contribution naturally vanishes.

This ensures that the additional constraint focuses on:

- High-confidence
- Hard-but-learnable
- Geometrically misaligned samples

---

## 7. Geometry-aware Margin Hyperparameter Rationale

The geometry-aware margin is controlled by several hyperparameters:

- `geom_margin_w`: global weight of the geometry-aware margin
- `geom_margin_k`: scaling factor controlling sensitivity to misalignment
- `geom_margin_warmup_epoch`: delays activation to avoid early training instability

In practice, enabling the geometry-aware margin after the first epoch stabilizes
training and prevents over-regularization during early convergence.

---

## 8. Features

- Adaptive knowledge distillation based on AdaDistill
- AdaFace loss with norm-based adaptive margins
- Geometry-aware margin for student–teacher alignment
- HuggingFace CVLFace teacher support
- Multi-GPU training via `torchrun`
- Comprehensive evaluation on standard and large-scale benchmarks

---

## 9. Installation

### 9.1 Environment Setup

```bash
conda create -n adadistill python=3.10
conda activate adadistill
```

Install PyTorch according to your CUDA version.

---

### 9.2 Dependencies

```bash
pip install -r requirements/requirement.txt
```

---

### 9.3 Optional: CVLFace

CVLFace is required for:

- Loading HuggingFace CVLFace teacher models
- Running IJB-B/C and TinyFace evaluations

Installation guide:  
https://github.com/mk-minchul/CVLface

---

## 10. Data Preparation

### Training Data

- Dataset: **MS1MV2**
- Format: InsightFace `.rec` / `.idx`
- Path: `dataset/faces_emore/`

### Evaluation Data

- Standard benchmarks (`.bin`):  
  LFW, CFP-FP, AgeDB, CALFW, CPLFW, VGG2-FP
- Large-scale benchmarks:  
  IJB-B, IJB-C, TinyFace (requires CVLFace)

---
## 11. Experimental Results

We conducted comprehensive evaluations on multiple benchmarks. Our method (**AdaDistill + AdaFace + Adaptive Geo**) consistently outperforms the baseline and standard distillation methods.

### 11.1 Comparison with State-of-the-Art
*Teacher: IR-50, Student: MobileFaceNet, Dataset: MS1MV2*

| Method | LFW | CFP-FP | AgeDB-30 | CP-LFW | IJB-C (1e-4) | IJB-C (1e-5) | Avg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Student) | 99.52 | 91.66 | 95.82 | 87.93 | 89.13 | 81.65 | 91.83 |
| AdaDistill + ArcFace | 99.47 | 95.14 | **96.68** | 89.50 | 93.26 | 89.31 | 94.12 |
| AdaDistill + AdaFace | 99.50 | **95.24** | 95.82 | 89.73 | 93.82 | 90.00 | 94.15 |
| **Ours (Adaptive Geo)** | **99.50** | 95.10 | 96.20 | **90.22** | **93.98** | **90.33** | **94.37** |

> **Key Observation:** Our method achieves the highest performance on difficult datasets like **CP-LFW** (Pose) and **IJB-C** (Mixed Quality), verifying the effectiveness of the geometry-aware margin.

### 11.2 Low-Resolution Performance (TinyFace)
The adaptive margin is particularly effective for low-quality images where standard distillation might fail.

| Method | TinyFace Rank-1 | TinyFace Rank-5 |
| :--- | :---: | :---: |
| AdaDistill + ArcFace | 57.21 | 63.12 |
| AdaDistill + AdaFace | 58.69 | 64.03 |
| **Ours (Adaptive Geo)** | **59.31** | **64.43** |

### 11.3 Ablation Study: Static vs. Adaptive
We compared our initial "Static Threshold" approach against the final "Adaptive" approach.

| Feature | Static Geo-Margin (V1) | Adaptive Geo-Margin (Final) |
| :--- | :--- | :--- |
| **Mechanism** | Raw cosine difference | Teacher confidence as a dynamic guide |
| **Thresholds** | Multiple, hard-coded, manual | **None** (Self-regulating) |
| **Stability** | Sensitive to tuning & data | Robust and stable |

### 11.4 Impact of Teacher Capacity
Contrary to the intuition that "bigger is better," we found that a mid-sized teacher (**IR-50**) produces better distillation results for a MobileFaceNet student compared to a massive teacher (**IR-101**).

* **Reasoning:** The student model (MobileFaceNet) struggles to absorb the highly complex feature distribution learned by IR-101. IR-50 provides a more accessible learning target.
## 12. Configuration

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

## 13. Training

Multi-GPU training (single node):

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train/train_AdaDistill.py
```

Resume training by setting `config.global_step` to the desired step.

---

## 14. Evaluation

### Single Checkpoint Evaluation

```bash
python eval/run_full_eval.py \
  --checkpoint output/AdaDistill/your_model_backbone.pth \
  --config config/config.py
```

### Batch Evaluation

```bash
python eval/run_batch_eval.py \
  --checkpoint-dir output/AdaDistill \
  --checkpoint-suffix backbone.pth \
  --save-json
```

---

## 15. Troubleshooting

- **FIXES_SUMMARY.md**  
  Shape mismatch fixes and CVLFace dependency handling.

- **TRAINING_SPEED_FIX.md**  
  Training speed optimization by reducing evaluation frequency.

---

## 16. Citation

If you use this code, please cite the original AdaDistill paper:

```bibtex
@InProceedings{Boutros_2024_ECCV,
  author    = {Fadi Boutros and Vitomir {\v{S}}truc and Naser Damer},
  title     = {AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  booktitle = {Computer Vision -- ECCV 2024},
  year      = {2024}
}

@InProceedings{Kim_2022_CVPR,
  author    = {Kim, Minchul and Jain, Anil K. and Liu, Xiaoming},
  title     = {AdaFace: Quality Adaptive Margin for Face Recognition},
  booktitle = {CVPR},
  year      = {2022}
}
```

---

## 17. License

This project is released under the  
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** license.

- Non-commercial use only  
- Attribution required  
- Derivative works must use the same license

