from dataloader import MultiDataset, SingleDataset, DKDataset, LesionDataset
from torch.utils.data import DataLoader
from BrainSeg import BrainSeg
import torch
from tqdm import tqdm
import torch.nn.functional as F
import nibabel as nib
import os
import argparse
import numpy as np
from BCLIP import BCLIP


def compute_dice(x, y, dim='2d'):
    if dim == '2d':
        # input size (B, C, H, W)
        dice = (2 * (x * y).sum([2, 3]) / (x.sum([2, 3]) + y.sum([2, 3]))).mean(-1)
    elif dim == '3d':
        # input size (B, C, L, W, H)
        dice = (2 * (x * y).sum([2, 3, 4]) / (x.sum([2, 3, 4]) + y.sum([2, 3, 4]))).mean(-1)
    return dice.item()


def eval_tissue_seg(config):
    if config.flag == 'multi':
        test_dataset = MultiDataset(texts_path=config.texts_path, images_path=config.images_path, index='Sheet1')
    elif config.flag == 'single':
        test_dataset = SingleDataset(texts_path=config.texts_path, images_path=config.images_path, modality=config.modality)
    else:
        raise ValueError(f"flag must be 'multi' or 'single', but got: '{config.flag}'")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    MyCLIP = BCLIP(in_channels=6).to(config.device)
    MyCLIP.load_state_dict(torch.load(os.path.join(config.clip_dir, 'BCLIP.pt'), map_location=config.device)['model_state_dict'])
    model = BrainSeg(img_size=config.img_size, in_channels=config.in_channels, out_channel=config.out_channels, feature_size=48, use_checkpoint=True).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'BrainSeg_tissue.pt'), map_location=torch.device(config.device))['model_state_dict'])
    model.eval()
    dice_scores = {0: [], 1: [], 2: []}
    all_dice_values = []
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for idx, data in enumerate(test_loader):
                image, target, text, affine, ID = data['images'], data['tissue'], data['text'], data['affine'], data['ID'][0]
                image = image.float().to(config.device)
                text = text.to(config.device)
                target = target.squeeze(1).long().to(config.device)
                text_feature, _ = MyCLIP.text_encoder(text)
                seg_out = model(image, text_feature)
                seg_out = torch.argmax(seg_out, dim=1)
                pred_seg = seg_out.squeeze(0).cpu().detach().numpy().astype(np.int32)
                affine = affine.squeeze(0).cpu().detach().numpy()
                pred_seg = nib.Nifti1Image(pred_seg, affine)
                os.makedirs(os.path.join(config.predir, ID), exist_ok=True)
                nib.save(pred_seg, os.path.join(config.predir, ID, f'pred_tissue.nii.gz'))
                seg_out = F.one_hot(seg_out, num_classes=config.out_channels).permute(0, 4, 1, 2, 3)[:, 1:]
                target = F.one_hot(target, num_classes=config.out_channels).permute(0, 4, 1, 2, 3)[:, 1:]
                dice_values = []
                for class_id in [0, 1, 2]:
                    dice = compute_dice(seg_out[:, class_id:class_id + 1], target[:, class_id:class_id + 1], '3d')
                    dice_values.append(dice)
                    dice_scores[class_id].append(dice)
                avg_dice_image = np.mean(dice_values)
                all_dice_values.append(avg_dice_image)
                pbar.update()
                print(f"{ID} - CSF Dice: {dice_values[0]:.4f} | "
                      f"GM Dice: {dice_values[1]:.4f} | "
                      f"WM Dice: {dice_values[2]:.4f} | "
                      f"Average Dice: {avg_dice_image:.4f}")
        avg_dice_csf = np.mean(dice_scores[0])
        avg_dice_gray_matter = np.mean(dice_scores[1])
        avg_dice_white_matter = np.mean(dice_scores[2])
        std_dice_csf = np.std(dice_scores[0])
        std_dice_gray_matter = np.std(dice_scores[1])
        std_dice_white_matter = np.std(dice_scores[2])
        avg_dice_all = np.mean(all_dice_values)
        std_dice_all = np.std(all_dice_values)
        print("\nOverall statistics:")
        print(f"CSF DICE - Mean: {avg_dice_csf:.4f}, Std: {std_dice_csf:.4f}")
        print(f"Gray Matter DICE - Mean: {avg_dice_gray_matter:.4f}, Std: {std_dice_gray_matter:.4f}")
        print(f"White Matter DICE - Mean: {avg_dice_white_matter:.4f}, Std: {std_dice_white_matter:.4f}")
        print(f"Overall Average DICE (CSF, GM, WM) - Mean: {avg_dice_all:.4f}, Std: {std_dice_all:.4f}")


def eval_dk_seg(config):
    test_dataset = DKDataset(texts_path=config.texts_path, images_path=config.images_path, index='Sheet1')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    MyCLIP = BCLIP(in_channels=6).to(config.device)
    MyCLIP.load_state_dict(torch.load(os.path.join(config.clip_dir, 'BCLIP.pt'), map_location=config.device)['model_state_dict'])
    model = BrainSeg(img_size=config.img_size, in_channels=config.in_channels, out_channel=config.out_channels, feature_size=48, use_checkpoint=True).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'BrainSeg_parc.pt'), map_location=torch.device(config.device))['model_state_dict'])
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for idx, data in enumerate(test_loader):
                image, target, text, affine, ID = data['images'], data['dk-struct'], data['text'], data['affine'], data['ID'][0]
                image = image.float().to(config.device)
                text = text.to(config.device)
                target = target.squeeze(1).long().to(config.device)
                text_feature, _ = MyCLIP.text_encoder(text)
                seg_out = model(image, text_feature)
                seg_out = torch.argmax(seg_out, dim=1)
                pred_seg = seg_out.squeeze(0).cpu().detach().numpy().astype(np.int32)
                affine = affine.squeeze(0).cpu().detach().numpy()
                pred_seg = nib.Nifti1Image(pred_seg, affine)
                os.makedirs(os.path.join(config.predir, ID), exist_ok=True)
                nib.save(pred_seg, os.path.join(config.predir, ID, f'pred_dk-struct.nii.gz'))
                seg_out = F.one_hot(seg_out, num_classes=config.out_channels).permute(0, 4, 1, 2, 3)[:, 1:]
                target = F.one_hot(target, num_classes=config.out_channels).permute(0, 4, 1, 2, 3)[:, 1:]
                dice_scores = []
                for roi in range(106):
                    seg_out_roi = seg_out[:, roi:roi + 1]
                    target_roi = target[:, roi:roi + 1]
                    dice_roi = compute_dice(seg_out_roi, target_roi, '3d')
                    dice_scores.append(dice_roi)
                dice_scores_str = ', '.join([f"ROI {roi + 1} Dice: {dice:.4f}" for roi, dice in enumerate(dice_scores)])
                pbar.update()
                print(f"{ID}: {dice_scores_str}")
                print(f"Mean Dice score (average of all ROIs): {np.mean(dice_scores):.4f}")


def eval_lesion_seg(config):
    test_dataset = LesionDataset(texts_path=config.texts_path, images_path=config.images_path, index='Sheet2')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    MyCLIP = BCLIP(in_channels=6).to(config.device)
    MyCLIP.load_state_dict(torch.load(os.path.join(config.clip_dir, 'BCLIP.pt'), map_location=config.device)['model_state_dict'])
    model = BrainSeg(img_size=config.img_size, in_channels=config.in_channels, out_channel=config.out_channels, feature_size=48, use_checkpoint=True).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'BrainSeg_lesion.pt'), map_location=torch.device(config.device))['model_state_dict'])
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for idx, data in enumerate(test_loader):
                image, text, affine, ID = data['images'], data['text'], data['affine'], data['ID'][0]
                image = image.float().to(config.device)
                text = text.to(config.device)
                text_feature, _ = MyCLIP.text_encoder(text)
                seg_out = model(image, text_feature)
                seg_out = torch.argmax(seg_out, dim=1)
                pred_seg = seg_out.squeeze(0).cpu().detach().numpy().astype(np.int32)
                affine = affine.squeeze(0).cpu().detach().numpy()
                pred_seg = nib.Nifti1Image(pred_seg, affine)
                os.makedirs(os.path.join(config.predir, ID), exist_ok=True)
                nib.save(pred_seg, os.path.join(config.predir, ID, f'pred_lesion.nii.gz'))
                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--texts_path', type=str, default=r'./test.xlsx', help='Data excel path')
    parser.add_argument('--images_path', type=str, default=r'./Sample', help='Data path')
    parser.add_argument('--img_size', type=int, nargs=3, default=(224, 256, 224), help='input image size')
    parser.add_argument('--in_channels', type=int, default=6, help='input channel')
    parser.add_argument('--out_channels', type=int, default=4, help='output channel')
    parser.add_argument('--device', type=str, default='cuda', help='GPU or CPU')
    parser.add_argument('--model_dir', type=str, default=r'/your/path/for/BrainSeg_model', help='path to the model')
    parser.add_argument('--clip_dir', type=str, default=r'/your/path/for/BCLIP', help='path to the B-CLIP model')
    parser.add_argument('--predir', type=str, default=r'./Sample', help='path to save results')
    parser.add_argument('--mode', type=str, default='tissue', help='which task you want to predict')
    parser.add_argument('--flag', type=str, default='multi', help='number of input modality')
    parser.add_argument('--modality', type=str, default='CT-brain.nii.gz', help='input modality')

    config = parser.parse_args()

    if config.mode == 'tissue':
        eval_tissue_seg(config)
    elif config.mode == 'dk':
        eval_dk_seg(config)
    elif config.mode == 'lesion':
        eval_lesion_seg(config)
    else:
        raise ValueError(f"mode must be 'tissue', 'dk' or 'lesion', but got: '{config.mode}'")