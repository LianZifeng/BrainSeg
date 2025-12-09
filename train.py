from datetime import datetime
import logging
import os
import argparse
parser = argparse.ArgumentParser()
config = parser.parse_args()
config.logs = './SegCLIPro_logs/'
date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
os.makedirs(config.logs, exist_ok=True)
logging.basicConfig(filename=os.path.join(config.logs, f"{date_str}.log"), level=logging.INFO, format='%(asctime)s %(message)s')

from dataloader import SegDataset
from torch.utils.data import DataLoader
from BrainSeg import BrainSeg
import torch
import math
import warnings
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from loss import DiceFocalLoss3D
from BCLIP.BCLIP import BCLIP
from torch.nn import functional as F


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


def compute_dice(x, y, dim='2d'):
    if dim == '2d':
        # input size (B, C, H, W)
        dice = (2 * (x * y).sum([2, 3]) / (x.sum([2, 3]) + y.sum([2, 3]))).mean(-1)
    elif dim == '3d':
        # input size (B, C, L, W, H)
        dice = (2 * (x * y).sum([2, 3, 4]) / (x.sum([2, 3, 4]) + y.sum([2, 3, 4]))).mean(-1)
    return dice.item()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def trainer(config):
    # Load data
    logging.info("Load dataset ...")
    dataset = SegDataset(texts_path=config.texts_path, images_path=config.images_path, mode_prob=0.5)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataset = SegDataset(texts_path=config.valid_texts_path, images_path=config.images_path, mode_prob=0.5)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    # Load model
    MyCLIP = BCLIP(in_channels=6).to(config.device)
    MyCLIP.load_state_dict(torch.load(os.path.join(config.clip_path, 'BCLIP.pt'), map_location=config.device)['model_state_dict'])
    MyCLIP.text_encoder.freeze()
    model = BrainSeg(img_size=config.img_size, in_channels=config.in_channels, out_channel=config.out_channels, feature_size=48, use_checkpoint=True).to(config.device)
    # Use B-Syn pre-trained model?
    if config.load_pretrained:
        model.load_state_dict(torch.load(config.pretrained_pth, map_location=config.device), strict=False)
        logging.info("Using pretrained weights")
    # Optimizer parameter settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim_lr, weight_decay=config.reg_weight)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_start_lr=1e-4, warmup_epochs=config.warmup_epochs, max_epochs=config.max_epochs)
    scaler = GradScaler()
    Loss = DiceFocalLoss3D(n_classes=config.out_channels, softmax=True)
    best_loss = float('inf')
    best_dice = -1
    os.makedirs(config.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.logdir)
    start_epoch = 0
    if config.checkpoint:
        checkpoint_path = os.path.join(config.model_dir, config.tag, '.pt')
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        best_dice = checkpoint.get('best_dice', -1)
        logging.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
    # Star training
    logging.info("Start training ...")
    lr_meter = AverageMeter()
    for epoch in tqdm(range(start_epoch, config.max_epochs), desc='Training'):
        model.train()
        loss_meter = AverageMeter()
        lr_meter.reset()
        for idx, data in enumerate(loader):
            image, target, text = data['images'], data['tissue'], data['text']
            image = image.float().to(config.device)
            target = target.squeeze(1).long().to(config.device)
            text = text.to(config.device)
            text_feature, _ = MyCLIP.text_encoder(text)
            optimizer.zero_grad()
            seg_out = model(image, text_feature)
            loss = Loss(seg_out, target)
            loss_meter.update(loss.item(), image.size(0))
            logging.info(f'epoch: [{epoch}/{config.max_epochs - 1}], batch: [{idx}/{len(loader) - 1}], '
                         f'current loss: {loss_meter.val}, current lr: {lr_meter.val}')
            current_lr = optimizer.param_groups[0]["lr"]
            lr_meter.update(current_lr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if (idx + 1) % 1000 == 0:
                os.makedirs(config.model_dir + config.tag, exist_ok=True)
                checkpoint_path = os.path.join(config.model_dir, config.tag, f'{epoch}_{idx}.pt')
                torch.save({
                    'epoch': epoch,
                    'iteration': idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss_meter.avg,
                }, checkpoint_path)
                logging.info(f"Checkpoint saved at {checkpoint_path}")
        scheduler.step()
        logging.info('----------------------------------------------------------------------------------------------')
        logging.info(f'epoch [{epoch}] finished with average loss: {loss_meter.avg}, average lr: {lr_meter.avg}')
        logging.info('----------------------------------------------------------------------------------------------')
        writer.add_scalar("train_loss", loss_meter.avg, epoch)
        writer.add_scalar("learning_rate", lr_meter.avg, epoch)
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_loss,
            }, config.model_dir + config.tag + '/' + 'best_loss.pt')
            logging.info(f"Best model saved with avg Dice: {best_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            logging.info('----------------------------------------validation----------------------------------------')
            model.eval()
            with torch.no_grad():
                dice_meter_csf = AverageMeter()
                dice_meter_gm = AverageMeter()
                dice_meter_wm = AverageMeter()
                # num_rois = 106
                # total_dice_scores = []
                for idx, data in enumerate(tqdm(valid_loader, desc='Validation')):
                    image, text, target = data['images'], data['text'], data['tissue']
                    image = image.float().to(config.device)
                    text = text.to(config.device)
                    target = target.squeeze(1).long().to(config.device)
                    text_feature, _  = MyCLIP.text_encoder(text)
                    seg_out = model(image, text_feature)
                    seg_out = torch.argmax(seg_out, dim=1)
                    seg_out = F.one_hot(seg_out, num_classes=config.out_channels).permute(0, 4, 1, 2, 3)[:, 1:]
                    target = F.one_hot(target, num_classes=config.out_channels).permute(0, 4, 1, 2, 3)[:, 1:]
                    dice_csf = compute_dice(seg_out[:, 0:1], target[:, 0:1], '3d')
                    dice_gm = compute_dice(seg_out[:, 1:2], target[:, 1:2], '3d')
                    dice_wm = compute_dice(seg_out[:, 2:3], target[:, 2:3], '3d')
                    dice_meter_csf.update(dice_csf)
                    dice_meter_gm.update(dice_gm)
                    dice_meter_wm.update(dice_wm)
                    logging.info(f"{test_dataset.data.patient_id[idx]} CSF Dice: {dice_csf:.4f}, GM Dice: {dice_gm:.4f}, WM Dice: {dice_wm:.4f}")
                    # dice_scores = []
                    # for roi in range(num_rois):
                    #     seg_out_roi = seg_out[:, roi:roi + 1]
                    #     target_roi = target[:, roi:roi + 1]
                    #     dice_roi = compute_dice(seg_out_roi, target_roi, '3d')
                    #     dice_scores.append(dice_roi)
                    # mean_dice = np.mean(dice_scores)
                    # total_dice_scores.append(mean_dice)
                    # for roi, dice in enumerate(dice_scores):
                    #     logging.info(f"ROI {roi + 1} Dice score: {dice:.4f}")
                    # logging.info(f"Mean Dice score (average of all ROIs): {mean_dice:.4f}")
                logging.info(f"CSF Dice score (avg): {dice_meter_csf.avg:.4f}")
                logging.info(f"Gray Matter Dice score (avg): {dice_meter_gm.avg:.4f}")
                logging.info(f"White Matter Dice score (avg): {dice_meter_wm.avg:.4f}")
                avg_dice = (dice_meter_csf.avg + dice_meter_gm.avg + dice_meter_wm.avg) / 3.0
                # avg_dice = np.mean(total_dice_scores)
                logging.info(f"Mean Dice score (average of all data with all ROIs): {avg_dice:.4f}")
                logging.info('--------------------------------------------------------------------------------------')
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': loss_meter.avg,
                        'best_dice': best_dice,
                    }, os.path.join(config.model_dir + config.tag, 'best_dice.pt'))
                    logging.info(f"Best model saved with avg Dice: {best_dice:.4f}")
    writer.close()

if __name__ == '__main__':
    config.texts_path = '/your/path/to/texts_path.xlsx'
    config.images_path = '/your/path/to/BrainCLIP/images_path'
    config.batch_size = 1
    config.img_size = (224, 256, 224)
    config.in_channels = 6
    config.out_channels = 4
    config.optim_lr = 1e-4
    config.reg_weight = 1e-5
    config.warmup_epochs = 1
    config.max_epochs = 32
    config.tag = '001'
    config.logdir = './SegCLIPro_logdir/' + config.tag
    config.load_pretrained = True
    config.pretrained_pth = '/your/path/to/pretrained_pth.pth'
    config.checkpoint = False
    config.model_dir = './SegCLIPro model/'
    config.device = 'cuda'
    config.valid_texts_path = '/your/path/to/valid_texts_path.xlsx'
    config.clip_path = '/your/path/to/BCLIP.pt'
    trainer(config)