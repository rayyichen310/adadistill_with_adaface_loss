


```
# AdaDistill: Adaptive Knowledge Distillation with AdaFace Loss and Geometry-aware Margin

This repository provides a **PyTorch implementation of AdaDistill** for deep
face recognition, extended with **AdaFace loss** and a **geometry-aware margin**
to improve student–teacher feature alignment during knowledge distillation.

The framework supports **HuggingFace-based teacher models**, **multi-GPU
training via `torchrun`**, and **comprehensive evaluation pipelines** across
both standard and large-scale face recognition benchmarks.

---

## 1. Overview

AdaDistill is an adaptive knowledge distillation framework designed for
margin-based face recognition models.  
This project extends the original AdaDistill method by:

1. Integrating **AdaFace loss** as the classification head.
2. Introducing a **geometry-aware margin** that explicitly considers the
   geometric relationship between student and teacher embeddings.

These extensions aim to improve distillation stability, convergence speed,
and final recognition performance.

---

## 2. Key Components

### 2.1 AdaFace-based Adaptive Margin

AdaFace uses the **L2 norm of the embedding** as a proxy for image quality.

- High-norm embeddings (high-quality samples) receive larger angular margins.
- Low-norm embeddings (low-quality or ambiguous samples) receive smaller margins.

This prevents the student from overfitting to noisy or low-quality samples
during distillation and encourages more robust feature learning.

---

### 2.2 Geometry-aware Distillation Margin

In addition to norm-based adaptation, this project introduces a
**geometry-aware margin** that focuses on student–teacher alignment.

For each sample, the following similarities are considered:

- Student–Teacher similarity:  
  `cos(f_s, f_t)`
- Teacher confidence for the ground-truth class:  
  `cos(w_y, f_t)`

If the teacher is confident about the class but the student embedding is still
poorly aligned with the teacher, an additional margin penalty is applied.

This mechanism:
- Emphasizes informative and learnable samples.
- Encourages the student feature space to better match the teacher’s geometry.
- Avoids unnecessary penalties on already aligned samples.

---

## 3. Features

- **Adaptive Knowledge Distillation** based on AdaDistill.
- **AdaFace loss integration** with norm-based adaptive margins.
- **Geometry-aware margin** for student–teacher alignment.
- **HuggingFace teacher support** (CVLFace models).
- **Multi-GPU training** via `torchrun`.
- **Comprehensive evaluation** on standard and large-scale benchmarks.

---

## 4. Installation

### 4.1 Environment Setup

```bash
conda create -n adadistill python=3.10
conda activate adadistill
