"""
Code for B-CLIP training
"""

from dataloader import CLIPDataset
from torch.utils.data import DataLoader
from BCLIP import BCLIP
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
import numpy as np
import os
from torch.cuda.amp import GradScaler
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter


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


class ClipLoss(nn.Module):

    def __init__(self, cache_labels=True):
        super().__init__()
        self.cache_labels = cache_labels

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def trainer(config):
    # Set logging
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    os.makedirs(config.logs, exist_ok=True)
    logging.basicConfig(filename=config.logs + date_str + '.log', level=logging.INFO, format='%(asctime)s %(message)s')
    # Load data
    logging.info("Load dataset ...")
    dataset = CLIPDataset(texts_path=config.texts_path, images_path=config.images_path, parameter_path=config.parameter_path,
                          lesion_path=config.lesion_path, lesion_parameter_path=config.lesion_parameter_path)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # Load model
    logging.info("Initalize model ...")
    MyCLIP = BrainCLIP(in_channels=config.in_channels).to(config.device)
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    # If there are multiple GPUs, use DataParallel
    if num_gpus > 1:
        MyCLIP = nn.DataParallel(MyCLIP).to(config.device)
    # Optimizer parameter settings
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(MyCLIP.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": config.wd},
        ],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    scaler = GradScaler()
    total_steps = (len(loader) // config.accum_freq) * config.epochs
    scheduler = cosine_lr(optimizer, config.lr, config.warmup, total_steps)
    # Loss
    loss = ClipLoss()
    best_loss = float('inf')
    os.makedirs(config.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=config.logdir)
    # Load checkpoint
    start_epoch = 0
    if config.checkpoint:
        checkpoint_path = os.path.join(config.model_dir, config.tag, '.pt')
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model_state_dict = checkpoint['model_state_dict']
        if next(MyCLIP.parameters()).device != config.device:
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        MyCLIP.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        logging.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
    # Star training
    logging.info("Start training ...")
    lr_meter = AverageMeter()
    for epoch in tqdm(range(start_epoch, config.epochs)):
        MyCLIP.train()
        num_batches_per_epoch = len(loader) // config.accum_freq
        loss_meter = AverageMeter()
        lr_meter.reset()
        for idx, data in enumerate(loader):
            i_accum = idx // config.accum_freq
            step = num_batches_per_epoch * epoch + i_accum
            scheduler(step)
            current_lr = optimizer.param_groups[0]["lr"]
            lr_meter.update(current_lr)
            image, text = data['images'], data['text']
            image = image.float().to(config.device)
            text = text.to(config.device)
            optimizer.zero_grad()
            image_features, text_features, logit_scale = MyCLIP(image, text)
            if isinstance(MyCLIP, nn.DataParallel):
                logit_scale = logit_scale.mean()
            losses = loss(image_features, text_features, logit_scale, output_dict=True)
            total_loss = sum(losses.values())
            loss_meter.update(total_loss.item(), text.size(0))
            logging.info(f'epoch [{epoch}/{config.epochs - 1}], batch [{idx}/{len(loader) - 1}], '
                         f'current loss: {loss_meter.val}, current lr: {current_lr}')
            backward(total_loss, scaler)
            scaler.step(optimizer)
            scaler.update()
        logging.info('----------------------------------------------------------------------------------------------')
        logging.info(f'epoch [{epoch}] finished with average loss: {loss_meter.avg}, average lr: {lr_meter.avg}')
        logging.info('----------------------------------------------------------------------------------------------')
        writer.add_scalar("train_loss", loss_meter.avg, epoch)
        writer.add_scalar("learning_rate", lr_meter.avg, epoch)
        if loss_meter.avg < best_loss:
            os.makedirs(config.model_dir + config.tag, exist_ok=True)
            best_loss = loss_meter.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': MyCLIP.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_loss,
            }, config.model_dir + config.tag + '/' + str(epoch) + '_epochs.pt')
    torch.save({'model_state_dict': MyCLIP.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': loss_meter.avg,
                }, config.model_dir + config.tag + '/' + config.tag + '.pt')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.texts_path = r'/your/path/to/texts_path.xlsx'
    config.images_path = r'/your/path/to/images_path'
    config.parameter_path = r'/your/path/to/parameter_path.xlsx'
    config.lesion_path = r'/your/path/to/lesion_path'
    config.lesion_parameter_path = r'/your/path/to/lesion_parameter_path.xlsx'
    config.batch_size = 128
    config.in_channels = 6
    config.device = 'cuda'
    config.lr = 0.0001
    config.epochs = 100
    config.model_dir = '/your/path/to/save/model'
    config.tag = '001'
    config.accum_freq = 1
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.eps = 1e-08
    config.wd = 0.2
    config.warmup = 1600
    config.checkpoint = True
    config.logs = '/your/path/to/save/log'
    config.logdir = '/your/path/to/save/logdir' + config.tag
    trainer(config)