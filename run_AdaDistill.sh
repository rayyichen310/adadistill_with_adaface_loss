#!/usr/bin/env bash
set -e

# Adjust GPU list and process count to match your machine.
export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train/train_AdaDistill.py

# 如果需要自動清理殘留的訓練進程，取消下行註解：
# ps -ef | grep "train_AdaDistill.py" | grep -v grep | awk '{print "kill -9 "$2}' | sh
