# AdaDistill：結合 AdaFace Loss 與幾何感知蒸餾的臉部辨識框架（中文說明）

本文件提供本專案的中文概要說明，著重在專案特色、AdaFace 與幾何感知蒸餾（geometry-aware KD）的設計概念，以及基本使用方式與授權條款。若需更完整的指令與實作細節，請一併參考英文版 `Readme.md`。

---

## 1. 專案特色概覽

本專案實作並擴充了 **AdaDistill** 演算法，用於深度臉部辨識（deep face recognition），主要特點如下：

- **AdaFace-based classification head 與自適應蒸餾**
  - 在原始 AdaDistill 結構上，將分類頭替換為 **AdaFace** 族群的 margin-based loss。
  - 利用特徵向量的 L2 norm 作為樣本品質的 proxy，對 margin 進行自適應調整：高品質樣本給予較大 margin，低品質樣本則降低 margin，以減少過度懲罰。

- **幾何感知的知識蒸餾（Geometry-aware KD Margin）**
  - 在 AdaFace 的 norm-based margin 之外，加入一個與 **student–teacher 特徵幾何差異** 相關的懲罰項。
  - 當 teacher 對某類別具有高信心，而 student 與 teacher 的特徵相似度仍顯著偏低時，該樣本會得到額外的角度 margin，促使 student 優先縮小這部分差距。

- **整合 HuggingFace CVLFace teacher 模型【需安裝 CVLface】**
  - 當 `config.teacher` 設為 `cvlface_ir50` 或 `cvlface_ir101` 時，`train/train_AdaDistill.py` 會透過  
    `CVLface.cvlface.general_utils.huggingface_model_utils.load_model_by_repo_id`  
    從 HuggingFace Hub 載入對應的 CVLFace 預訓練模型（如 `minchul/cvlface_adaface_ir50_webface4m`）。
  - 使用前需：
    - 正確安裝並設定 `CVLface/` 專案及其依賴。
    - 設定 HuggingFace token（環境變數 `HF_TOKEN`）。
    - 在 `config/config.py` 中檢查 `config.teacher_repo_id` 與 `config.teacher_cache`。

- **多種評估流程支援【IJB / TinyFace 評估需安裝 CVLface】**
  - `.bin` 驗證資料集：LFW、CFP-FP、CFP-FF、AgeDB-30、CALFW、CPLFW、VGG2-FP。
  - IJB-B / IJB-C：由 `eval/ijb_evaluator.py` 呼叫 CVLface 的 `ijbbc.evaluate` 完成評估。
  - TinyFace：由 `eval/tinyface_evaluator.py` 呼叫 CVLface 的 `tinyface.evaluate` 完成評估。
  - 若尚未安裝 CVLface 或未準備相應 Arrow 資料，可在 `config/config.py` 中關閉 `config.eval_ijb` 與 `config.eval_tinyface`，僅執行 `.bin` 驗證。

- **訓練效率與穩定性優化**
  - 已修正原始程式中評估 callback 在每個訓練步驟都觸發所導致的效能問題，訓練速度得以恢復（詳見 `TRAINING_SPEED_FIX.md`）。
  - 在 `utils/losses.py` 中加入必要的 clamp 與 warmup 機制，以降低數值不穩定與 NaN 的風險；相關修正整理於 `FIXES_SUMMARY.md`。

---

## 2. AdaFace 與幾何感知蒸餾之原理概述

本節簡要說明 `utils/losses.py` 中 `AdaptiveAAdaFace` 類別所採用的兩個關鍵設計：AdaFace 的 norm-based margin，以及 geometry-aware KD margin。為方便閱讀，以下均使用文字符號表示，而不使用 LaTeX 標記。

### 2.1 AdaFace：基於特徵 norm 的自適應 margin

在 AdaFace 中，每一筆樣本有一個 embedding 向量 `f`，以及對應的 L2 norm `||f||`，可視為該樣本特徵品質的指標。訓練過程中會維護 running mean / std，對 `||f||` 做標準化與縮放，形成一個品質相關的縮放因子 `q`：

- 當 `||f||` 較大（高品質樣本）時，`q` 較大 → 實際使用的 margin 較大，類別分離更明顯。
- 當 `||f||` 較小（低品質樣本）時，`q` 較小甚至為負 → margin 變小，減少對不穩定樣本的過度懲罰。

透過這個設計，模型會自動將更多判別能力分配給高品質樣本，有助於提升整體穩定性與辨識性能。

### 2.2 Geometry-aware KD margin：關注 student–teacher 幾何差異

在蒸餾情境中，每個樣本會同時使用：

- student 特徵：`f_s`
- teacher 特徵：`f_t`
- 該樣本所屬類別 `y` 的權重向量：`w_y`

實作中會計算兩個關鍵量，反映 student 與 teacher 在特徵空間中的關係，以及 teacher 的自信程度：

1. `cos_st = cos(f_s, f_t)`  
   代表 student 與 teacher 特徵之間的 cosine 相似度，相近表示 student 已經較好地對齊 teacher。

2. `lam = cos(w_y, f_t)`  
   代表 teacher 特徵與該類別權重之間的 cosine，可視為 teacher 在該類別上的信心指標。

幾何懲罰項（記作 `geom_delta`）的核心概念可寫成類似：

```text
geom_delta ∝ max(0, lam - cos_st) * lam
```

直觀解讀如下：

- 當 `cos_st` 已接近 `lam` 時，student 與 teacher 在該類別上的表現差距很小，`lam - cos_st` 為零或接近零，此時幾何懲罰幾乎不作用。
- 當 `lam` 較大（teacher 對該類別很有把握），但 `cos_st` 偏低（student 跟得不好）時，`lam - cos_st` 為正且幅度較大，`geom_delta` 也會變大，對這些「teacher 高自信但 student 落後」的樣本施加更強的額外 margin。

在實作中，`geom_delta` 會再經由 `config.geom_margin_k` 等參數放大或縮小，並乘上總體權重 `config.geom_margin_w`；此外，`config.geom_margin_warmup_epoch` 控制幾何項在訓練前幾個 epoch 逐步引入，以避免訓練初期懲罰過強。

### 2.3 結合 norm-based 與 geometry-aware margin

在 `AdaptiveAAdaFace` 中，最終作用在 logit（或角度）上的 margin，可以概念化為：

```text
margin_scaler ≈ (norm-based 部分 q) + (geom_margin_w * geom_delta)
```

其中：

- norm-based 部分 `q` 來自 AdaFace，反映樣本本身的品質（透過 `||f||`）。
- 幾何部分 `geom_delta` 來自 geometry-aware KD，反映「teacher 很有把握但 student 尚未對齊」的程度。

這種組合讓模型在訓練時特別關注以下樣本：

- 特徵品質高（`||f_s||` 大），且
- teacher 對該類別信心高（`lam` 大），但
- student 與 teacher 差距仍大（`cos_st` 小）。

換言之，蒸餾過程會將更多資源集中在「關鍵且可學習的樣本」上，以提升學習效率與最終表現。

---

## 3. 環境與依賴（概要）

- **基本環境**
  - 建議使用 Python 3.10。
  - 安裝對應 GPU / CUDA 的官方 PyTorch 版本。
  - 在 `AdaDistill/` 目錄下執行：

    ```bash
    pip install -r requirements/requirement.txt
    ```

- **CVLface 相關依賴【僅在使用 CVLface 功能時需要】**
  - 若要使用 HuggingFace 上的 CVLFace teacher 模型（`config.teacher = "cvlface_ir50"` 或 `"cvlface_ir101"`），本專案會透過 `CVLface.cvlface.general_utils.huggingface_model_utils` 載入模型，因此需：
    - 安裝並設定 `CVLface/` 專案及其依賴。
    - 設定 HuggingFace token（環境變數 `HF_TOKEN`）。
    - 在 `config/config.py` 中確認 `config.teacher_repo_id` 與 `config.teacher_cache`。
  - 若要執行 IJB-B/C 或 TinyFace 評估，`eval/ijb_evaluator.py` 與 `eval/tinyface_evaluator.py` 亦依賴 CVLface 的 evaluation 模組。

- **未安裝 CVLface 的情況**
  - 可改用非 CVLface teacher（例如 `iresnet50` 等），並提供對應的本地預訓練權重。
  - 建議在 `config/config.py` 中關閉：

    ```python
    config.eval_ijb = False
    config.eval_tinyface = False
    ```

    僅執行 `.bin` 驗證流程。

---

## 4. 基本使用流程（高層概述）

### 4.1 準備資料

- 訓練資料：MS1MV2（faces_emore），包含 `train.rec` 與 `train.idx`，放置於：

  ```text
  dataset/faces_emore/train.rec
  dataset/faces_emore/train.idx
  ```

- 驗證資料：LFW、CFP、AgeDB 等 `.bin` 檔放置於同一資料夾，例如：

  ```text
  dataset/faces_emore/lfw.bin
  dataset/faces_emore/cfp_fp.bin
  ...
  ```

- 若要進行 IJB-B/C 或 TinyFace 評估，需額外準備 Arrow 格式資料並置於：

  ```text
  dataset/facerec_val/IJBB_gt_aligned/...
  dataset/facerec_val/IJBC_gt_aligned/...
  dataset/facerec_val/tinyface_aligned_pad_0.1/...
  ```

  並確保已安裝與設定 CVLface。

### 4.2 設定 `config/config.py`

在 `config/config.py` 中配置：

- 資料與輸出路徑：

  ```python
  config.dataset = "emoreIresNet"
  config.output = "output/AdaDistill_sync/"
  ```

- 網路與 loss：

  ```python
  config.network = "mobilefacenet"   # 或 iresnet50 / iresnet18 等
  config.loss = "AdaFace"
  ```

- 幾何感知蒸餾相關參數：

  ```python
  config.use_geom_margin = True
  config.geom_margin_w = 1.0
  config.geom_margin_k = 3.0
  config.geom_margin_warmup_epoch = 1
  ```

- teacher 設定：
  - 使用本地 iresnet teacher 時，指定 `config.teacher` 與 `config.pretrained_teacher_path`。
  - 使用 CVLface teacher 時，指定：

    ```python
    config.teacher = "cvlface_ir50"  # 或 "cvlface_ir101"
    config.teacher_repo_id = "minchul/cvlface_adaface_ir50_webface4m"
    config.teacher_cache = "~/.cvlface_cache/minchul/cvlface_adaface_ir50_webface4m"
    ```

### 4.3 啟動訓練

在 `AdaDistill/` 目錄下，可使用範例腳本啟動訓練：

```bash
bash run_AdaDistill.sh
```

該腳本預設使用：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train/train_AdaDistill.py
```

如需單機多卡訓練，可調整 `CUDA_VISIBLE_DEVICES` 與 `--nproc_per_node`，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train/train_AdaDistill.py
```

### 4.4 模型評估

- 單一 checkpoint 完整評估：

  ```bash
  python eval/run_full_eval.py \
    --checkpoint output/AdaDistill/your_model_backbone.pth \
    --config config/config.py \
    --val-data ./dataset/faces_emore \
    --ijb-root ./dataset/facerec_val \
    --tinyface-root ./dataset/facerec_val \
    --device cuda:0
  ```

  若未安裝 CVLface 或未準備 IJB/TinyFace 資料，可將 `eval_ijb` 與 `eval_tinyface` 設為 `False`，只執行 `.bin` 驗證。

- 多個 checkpoint 批次評估：

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

  會對指定資料夾中所有符合條件的 checkpoint 逐一評估，並將結果輸出為 CSV 及（選擇性）JSON。

---

## 5. 引用與授權

### 5.1 引用（Citation）

如果你在研究工作中使用本實作，請引用 AdaDistill 原始論文：

```bibtex
@InProceedings{Boutros_2024_ECCV,
  author    = {Fadi Boutros and Vitomir {\v{S}}truc and Naser Damer},
  title     = {AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  booktitle = {Computer Vision -- ECCV 2024},
  month     = {October},
  year      = {2024}
}
```

若同時使用本專案中整合的 CVLFace 預訓練模型或其評估工具，亦建議依 CVLface 官方說明文件補充相應引用，以尊重原作者之貢獻。

### 5.2 授權與使用限制

本 repo 是基於原始 AdaDistill 專案的延伸實作，原始專案授權為 **Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)**：

- 僅限 **非商業用途**。
- 需保留原作者與引用資訊。
- 修改後的作品需以相同授權方式釋出。

請在使用本程式碼時遵守上述授權條款；關於更詳細的法條內容，請參考原始 AdaDistill 專案的 LICENSE。實際具法律約束力的條文與解釋，應以 Creative Commons 官方文件及原始專案提供之授權聲明為準。

