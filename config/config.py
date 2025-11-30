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
config.output = "output/AdaDistillref/" # train model output folder

# teacher path / huggingface
config.pretrained_teacher_path = "output/teacher/295672backbone.pth"
config.pretrained_teacher_header_path = "output/teacher/295672header.pth" # teacher folder
config.teacher_repo_id = "minchul/cvlface_adaface_ir50_webface4m"
config.teacher_cache = "~/.cvlface_cache/minchul/cvlface_adaface_ir50_webface4m"

config.global_step=0 # step to resume

# Margin-penalty loss configurations
config.s=64.0
config.m=0.4
config.h=0.333
config.t_alpha=0.01

#AdaDistill configuration
config.adaptive_alpha=True




config.loss="AdaFace"  #  Option : ArcFace, CosFace, AdaFace, MLLoss

# type of network to train [iresnet100 | iresnet50 | iresnet18 | mobilefacenet]
config.network = "mobilefacenet"
config.teacher = "cvlface_ir50"

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
config.ijb_num_workers = 14
config.ijb_flip = True

# TinyFace evaluation configuration
config.eval_tinyface = True
config.tinyface_root = "./dataset/facerec_val"
config.tinyface_targets = ["tinyface_aligned_pad_0.1"]
config.tinyface_batch_size = 384
config.tinyface_num_workers = 14
config.tinyface_flip = True


if config.dataset == "emoreIresNet":
    config.rec = "./dataset/faces_emore"
    config.val_rec = "./dataset/faces_emore"
    config.db_file_format="rec"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw", "vgg2_fp"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

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
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_func
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
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_func
