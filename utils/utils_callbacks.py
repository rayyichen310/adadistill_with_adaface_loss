import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from eval.ijb_evaluator import run_ijb_evaluation
from eval.tinyface_evaluator import run_tinyface_evaluation


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112), writer=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.writer = writer
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            if self.writer is not None:
                prefix = f"val/{self.ver_name_list[i]}"
                self.writer.add_scalar(f"{prefix}/XNorm", xnorm, global_step)
                self.writer.add_scalar(f"{prefix}/Accuracy_Flip", acc2, global_step)
                self.writer.add_scalar(f"{prefix}/Accuracy_Highest", self.highest_acc_list[i], global_step)
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        # Frequency control is now handled in the training loop, not here
        if self.rank == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, epoch: int, target_logit_mean: float, lma:float, cos_theta_tmp:float, geom_penalty:float=0.0, geom_weighted:float=0.0):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                # TODO: resume time_total is not working
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('train/time_for_end_hours', time_for_end, global_step)
                    self.writer.add_scalar('train/loss', loss.avg, global_step)
                    self.writer.add_scalar('train/target_logit_mean', target_logit_mean, global_step)
                    self.writer.add_scalar('train/lma', lma, global_step)
                    self.writer.add_scalar('train/cos_theta_tmp', cos_theta_tmp, global_step)
                    self.writer.add_scalar('train/geom_penalty', geom_penalty, global_step)
                    self.writer.add_scalar('train/geom_weighted', geom_weighted, global_step)
                    self.writer.add_scalar('train/speed_samples_per_sec', speed_total, global_step)
                    self.writer.add_scalar('train/epoch', epoch, global_step)
                msg = "Speed %.2f samples/sec   Loss %.4f target_logit_mean %.4f lma %.4f  cos_theta_tmp %.4f geom_penalty %.4f geom_weighted %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss.avg, target_logit_mean, lma, cos_theta_tmp, geom_penalty, geom_weighted, epoch, global_step, time_for_end
                )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone: torch.nn.Module, header: torch.nn.Module = None):
        if global_step > 100 and self.rank == 0:
            torch.save(backbone.module.state_dict(), os.path.join(self.output, str(global_step)+ "backbone.pth"))
        if global_step > 100 and header is not None:
            torch.save(header.module.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))


class CallBackIJB(object):
    """IJB-B/C evaluation using Arrow datasets (HuggingFace) and CVLface metrics."""

    def __init__(self, frequent, rank, ijb_targets, ijb_root, device, batch_size=64, num_workers=4, flip=True, writer=None):
        self.frequent = frequent
        self.rank = rank
        self.ijb_targets = ijb_targets
        self.ijb_root = ijb_root
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip = flip
        self.writer = writer

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank != 0:
            return
        # Frequency control is now handled in the training loop, not here
        backbone.eval()
        with torch.no_grad():
            for name in self.ijb_targets:
                try:
                    result = run_ijb_evaluation(
                        backbone.module,
                        dataset_root=self.ijb_root,
                        dataset_name=name,
                        device=self.device,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        flip=self.flip,
                    )
                    logging.info(f"[IJB][{name}] {result}")
                    if self.writer is not None:
                        for k, v in result.items():
                            self.writer.add_scalar(f"IJB/{name}/{k}", v, num_update)
                except Exception as e:
                    logging.warning(f"IJB evaluation failed for {name}: {e}")
        backbone.train()


class CallBackTinyFace(object):
    """TinyFace evaluation using Arrow datasets and CVLface metrics."""

    def __init__(self, frequent, rank, targets, root, device, batch_size=64, num_workers=4, flip=True, writer=None):
        self.frequent = frequent
        self.rank = rank
        self.targets = targets
        self.root = root
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip = flip
        self.writer = writer

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank != 0:
            return
        # Frequency control is now handled in the training loop, not here
        backbone.eval()
        with torch.no_grad():
            for name in self.targets:
                try:
                    result = run_tinyface_evaluation(
                        backbone.module,
                        dataset_root=self.root,
                        dataset_name=name,
                        device=self.device,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        flip=self.flip,
                    )
                    logging.info(f"[TinyFace][{name}] {result}")
                    if self.writer is not None:
                        for k, v in result.items():
                            self.writer.add_scalar(f"TinyFace/{name}/{k}", v, num_update)
                except Exception as e:
                    logging.warning(f"TinyFace evaluation failed for {name}: {e}")
        backbone.train()
