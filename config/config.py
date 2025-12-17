from easydict import EasyDict as edict

config = edict()
config.dataset = "emoreIresNet" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 384

# batch size per GPU
config.lr = 0.1

# Saving path
config.output = "output/AdaDistill_sync/" # train model output folder (changed from AdaDistillref)

# teacher path / huggingface
config.pretrained_teacher_path = "output/teacher/295672backbone.pth" # Ignored when using HF model
config.pretrained_teacher_header_path = "output/teacher/295672header.pth" # Ignored when using HF model
config.teacher_repo_id = ""
config.teacher_cache = ""

config.global_step=0 # step to resume

# Margin-penalty loss configurations
config.s=64.0
config.m=0.4
config.h=0.333
config.t_alpha=0.01
# Geometry-aware KD margin (off by default)
config.use_geom_margin = True  # 如果要結合 (1 - cos(student, teacher)) 調整 margin，改成 True
config.geom_margin_w = 1.0      # 全局縮放因子 (Scale)，建議設為 1.0，讓模型自動決定
config.geom_margin_k = 2.0      # 幾何項縮放 k，控制 (1 - cos) 的量級
# config.geom_margin_mask = 0.8   # (已棄用) 自動根據 Teacher 信心決定
# config.geom_margin_baseline = 0.25  # (已棄用) 自動根據 Teacher 信心決定
config.geom_margin_warmup_epoch =  4 # 幾何項權重的 warmup epoch 數，0 表示不做 warmup

#AdaDistill configuration
config.adaptive_alpha=True




config.loss="AdaFace"  #  Option : ArcFace, CosFace, AdaFace, MLLoss

# type of network to train [iresnet100 | iresnet50 | iresnet18 | mobilefacenet]
config.network = "mobilefacenet"
config.teacher = "cvlface_ir101"

config.SE=False # SEModule

# optional: run a validation pass right after loading checkpoint
config.run_eval_at_start = False

# evaluation cadence
config.val_eval_every_n_epoch = 1  # lightweight val (lfw/cfp/agedb/...)
config.eval_every_n_epoch = 4

# IJB evaluation configuration
config.eval_ijb = True
config.ijb_root = "./dataset/facerec_val"
config.ijb_targets = ["IJBB_gt_aligned", "IJBC_gt_aligned"]
config.ijb_batch_size = 384
config.ijb_num_workers = 16
config.ijb_flip = True

# TinyFace evaluation configuration
config.eval_tinyface = True
config.tinyface_root = "./dataset/facerec_val"
config.tinyface_targets = ["tinyface_aligned_pad_0.1"]
config.tinyface_batch_size = 384
config.tinyface_num_workers = 16
config.tinyface_flip = True


if config.dataset == "emoreIresNet":
    config.rec = "./dataset/faces_emore"
    config.val_rec = "./dataset/faces_emore"
    config.db_file_format="rec"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  26
    # Warmup: 前 warmup_epoch 個 epoch 平滑將 lr 拉到正常值
    config.warmup_epoch = 1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw", "vgg2_fp"]
    config.eval_step=5686
    def lr_step_emore(epoch: int):
        # epoch 從 0 開始
        if epoch < config.warmup_epoch:
            # 用平滑二次函數暖機
            return ((epoch + 1) / (config.warmup_epoch + 1)) ** 2
        else:
            milestones = [8, 14, 20, 25]
            return 0.1 ** len([m for m in milestones if m - 1 <= epoch])
    config.lr_func = lr_step_emore

if config.dataset == "Idifface":
    config.rec = "./data/faces_emore"
    config.data_path="./dataset/Idifface"
    config.db_file_format="folder"

    config.num_classes = 10049
    config.num_image = 502450
    config.num_epoch = 60
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 982 * 4
    def lr_step_idif(epoch: int):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_idif
    config.sample = 50

if config.dataset == "CASIA_WebFace":
    config.rec = "./data/faces_webface_112x112"
    config.db_file_format="rec"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 60
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 3916
    def lr_step_casia(epoch: int):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_casia
