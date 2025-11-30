## AdaDistill with AdaFace Loss & HF Teacher

本資料夾包含 AdaDistill 的訓練與評估程式碼，並在原始 ECCV 2024 版本上做了以下擴充：

- 支援 **AdaFace loss** 的 adaptive distillation。
- 支援使用 HuggingFace 上的 **CVLFace teacher 模型**（例如 `cvlface_adaface_ir50_webface4m`、`cvlface_adaface_ir101_webface12m`）。
- 提供單次跑完所有驗證資料集的腳本（LFW/CFP/AgeDB/CALFW/CPLFW/VGG2-FP + IJBB/IJBC + TinyFace）。

---

## 環境安裝

1. 建立 Conda 環境（建議 Python 3.10）：

   ```bash
   conda create -n adadistill python=3.10
   conda activate adadistill
   ```

2. 安裝依賴：

   ```bash
   pip install -r requirements/requirement.txt
   ```

3. 如需從 HuggingFace 下載 teacher 模型（私有或需授權），請設定環境變數：

   ```bash
   export HF_TOKEN="你的 HuggingFace token"
   ```

---

## 數據準備

### 1. 訓練資料（MS1MV2）

- 下載 InsightFace 提供的 MS1MV2 資料集（112x112，rec 格式），解壓後放到：

  ```text
  dataset/faces_emore
  ```

- `config/config.py` 中預設：

  ```python
  config.dataset = "emoreIresNet"
  config.rec = "./dataset/faces_emore"
  config.val_rec = "./dataset/faces_emore"
  config.db_file_format = "rec"
  ```

### 2. 驗證資料（LFW / CFP / AgeDB / CALFW / CPLFW / VGG2-FP）

- 對應 `.bin` 檔放在：

  ```text
  dataset/faces_emore/*.bin
  ```

- 驗證清單在 `config/config.py`：

  ```python
  config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw", "vgg2_fp"]
  ```

### 3. IJB-B/C & TinyFace（Arrow 格式）

- 預期結構：

  ```text
  dataset/facerec_val/IJBB_gt_aligned/...
  dataset/facerec_val/IJBC_gt_aligned/...
  dataset/facerec_val/tinyface_aligned_pad_0.1/...
  ```

- 在 `config/config.py` 中：

  ```python
  config.eval_ijb = True
  config.ijb_root = "./dataset/facerec_val"
  config.ijb_targets = ["IJBB_gt_aligned", "IJBC_gt_aligned"]

  config.eval_tinyface = True
  config.tinyface_root = "./dataset/facerec_val"
  config.tinyface_targets = ["tinyface_aligned_pad_0.1"]
  ```

---

## 設定說明（`config/config.py`）

核心參數（只列出與訓練相關的主鍵）：

```python
from easydict import EasyDict as edict
config = edict()

# 資料集與輸出
config.dataset = "emoreIresNet"
config.output = "output/AdaDistillref/"

# 特徵與 loss
config.embedding_size = 512
config.loss = "AdaFace"      # ArcFace / CosFace / AdaFace / MLLoss
config.s = 64.0
config.m = 0.4
config.h = 0.333
config.t_alpha = 0.01
config.adaptive_alpha = True

# 學生網路
config.network = "mobilefacenet"   # 或 iresnet50 / iresnet18
config.SE = False                  # 使用 / 不使用 SE 模組

# Teacher（HuggingFace 模型）
config.teacher = "cvlface_ir50"    # "cvlface_ir50" 或 "cvlface_ir101" 等
config.teacher_repo_id = "minchul/cvlface_adaface_ir50_webface4m"
config.teacher_cache = "~/.cvlface_cache/minchul/cvlface_adaface_ir50_webface4m"

# 訓練控制
config.batch_size = 384
config.lr = 0.1
config.global_step = 0             # 0 = 從頭訓練；>0 = 從 checkpoint 繼續
```

若要從某個 step 繼續訓練（例如 `output/AdaDistillref/212282backbone.pth`）：

```python
config.output = "output/AdaDistillref/"
config.global_step = 212282
```

---

## 訓練方式

### 單卡訓練

預設腳本 `run_AdaDistill.sh` 使用 `torchrun` 單卡啟動：

```bash
cd /home/t2-503-3090ti/ray/adadistil/AdaDistill
sh run_AdaDistill.sh
```

腳本內容（簡化）：

```bash
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train/train_AdaDistill.py
```

如需多卡訓練，只要調整 `CUDA_VISIBLE_DEVICES` 和 `--nproc_per_node` 即可，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train/train_AdaDistill.py
```

---

## 評估：一次跑完所有驗證資料集

使用 `eval/run_full_eval.py` 對單一 checkpoint 做完整驗證：

```bash
cd /home/t2-503-3090ti/ray/adadistil/AdaDistill

PYTHONPATH=/home/t2-503-3090ti/ray/adadistil \
python eval/run_full_eval.py \
  --checkpoint output/AdaDistill/MFN_AdaArcDistill_backbone.pth \
  --val-data ./dataset/faces_emore \
  --ijb-root ./dataset/facerec_val \
  --tinyface-root ./dataset/facerec_val \
  --device cuda:0
```

這會輸出：

- LFW / CFP-FP / CFP-FF / AgeDB-30 / CALFW / CPLFW / VGG2-FP 的 Accuracy-Flip 與 XNorm。
- IJBB / IJBC 在多個 FPR 門檻下的 TPR（`tpr_at_fpr_x`）與對應 threshold。
- TinyFace 的 rank-1 / rank-5 / rank-20。

---

## 常見問題

- **為什麼一開始就跑 IJB/TinyFace？**  
  `config.run_eval_at_start = True` 時，會在訓練開始前先做一次驗證（包含 IJB/TinyFace），方便記錄 baseline。

- **`cos_theta_tmp` 接近 0.8 正常嗎？**  
  在 AdaFace + s=64, m=0.4 的設定下，訓練後期 cosθ 落在 0.7–0.85 很常見，重點是 loss 和驗證集 accuracy 是否持續改善。

- **MXNet / NumPy 的相容性警告**  
  專案在 `eval/verification.py` 中對 `np.bool` 做了 shim，以支援 NumPy >= 1.24；出現 FutureWarning 屬正常現象。

---

## 原始論文與授權

本專案基於 AdaDistill 原始程式碼修改而來；如在研究中使用，請同時引用：

```bibtex
@InProceedings{Boutros_2024_ECCV,
  author    = {Fadi Boutros, Vitomir {\v{S}}truc, Naser Damer},
  title     = {AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  booktitle = {Computer Vision - {ECCV} 2024 - 18th European Conference on Computer Vision},
  month     = {October},
  year      = {2024}
}
```

原始專案授權為 Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)，詳情請見上層目錄的 LICENSE。
