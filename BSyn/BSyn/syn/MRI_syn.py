import torch
from syn.transform import get_gauss
import torch.nn.functional as F
import syn.transform as transform
import random
from scipy.ndimage import gaussian_filter


def find_matching_file(age, meta_table, age_threshold, filename, modality_key):
    matching_files = meta_table[(meta_table['Modality'] == modality_key)]
    selected_row = matching_files.sample(n=1)
    syn_parameter = selected_row.iloc[0][3:].tolist()

    original_file = meta_table[(meta_table['Modality'] == "brain.nii.gz") & (meta_table['combined_col'] == filename)]
    original_parameter = original_file.iloc[0][3:].tolist()

    return syn_parameter,original_parameter


def GMM_mri_t1(brain_data, roi_data, tissue_data, age, age_threshold, meta_table, filename, modality_key, downsample=False):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max() - brain.min() + 1e-10)
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()

    syn_parameter, original_parmeter = find_matching_file(age, meta_table, age_threshold, filename, modality_key)

    index_all = torch.unique(tissue)
    gen_ima = tissue.clone()
    for ii in index_all:
        if ii != 0:
            mu1 = syn_parameter[int(ii) * 2]
            std1 = syn_parameter[int(ii) * 2 + 1]
            mu2 = original_parmeter[int(ii) * 2]
            std2 = original_parmeter[int(ii) * 2 + 1]

            struct_copy = brain.clone()
            struct_copy[tissue != int(ii)] = 0
            adjusted_img = (struct_copy - mu2) / std2 * std1 + mu1
            gen_ima[tissue == int(ii)] = adjusted_img[tissue == int(ii)]

    sigma = 0.8
    boundary_mask = torch.zeros_like(tissue, dtype=torch.bool)
    boundary_mask[:-1, :, :] |= (tissue[:-1, :, :] != tissue[1:, :, :])
    boundary_mask[:, :-1, :] |= (tissue[:, :-1, :] != tissue[:, 1:, :])
    boundary_mask[:, :, :-1] |= (tissue[:, :, :-1] != tissue[:, :, 1:])
    smoothed_img = gen_ima.clone()
    gauss_kernel = get_gauss(sigma).to(brain.device)
    smoothed_values = F.conv3d(smoothed_img, weight=gauss_kernel[None, None, ...], padding=1)
    smoothed_img[boundary_mask] = smoothed_values[boundary_mask]
    smoothed_img[tissue == 0] = smoothed_img[tissue != 0].min()

    gen_ima = transform.RandomBiasField()(smoothed_img)
    if downsample:
        gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()

    return gen_ima.squeeze()


def GMM_mri_t2(brain_data, roi_data, tissue_data, age, age_threshold, meta_table, filename, modality_key, downsample=False):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max() - brain.min() + 1e-10)
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()

    syn_parameter, original_parmeter = find_matching_file(age, meta_table, age_threshold, filename, modality_key)

    index_all = torch.unique(tissue)
    gen_ima = tissue.clone()
    for ii in index_all:
        if ii != 0:
            mu1 = syn_parameter[int(ii) * 2]
            std1 = syn_parameter[int(ii) * 2 + 1]
            mu2 = original_parmeter[int(ii) * 2]
            std2 = original_parmeter[int(ii) * 2 + 1]

            struct_copy = brain.clone()*-1
            struct_copy[tissue != int(ii)] = 0
            adjusted_img = (struct_copy + mu2) / std2 * std1 + mu1
            gen_ima[tissue == int(ii)] = adjusted_img[tissue == int(ii)]

    sigma = 0.8
    boundary_mask = torch.zeros_like(tissue, dtype=torch.bool)
    boundary_mask[:-1, :, :] |= (tissue[:-1, :, :] != tissue[1:, :, :])
    boundary_mask[:, :-1, :] |= (tissue[:, :-1, :] != tissue[:, 1:, :])
    boundary_mask[:, :, :-1] |= (tissue[:, :, :-1] != tissue[:, :, 1:])
    smoothed_img = gen_ima.clone()
    gauss_kernel = get_gauss(sigma).to(brain.device)
    smoothed_values = F.conv3d(smoothed_img, weight=gauss_kernel[None, None, ...], padding=1)
    smoothed_img[boundary_mask] = smoothed_values[boundary_mask]
    smoothed_img[tissue == 0] = smoothed_img[tissue != 0].min()

    gen_ima = transform.RandomBiasField()(smoothed_img)
    if downsample:
        gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    return gen_ima.squeeze()


def GMM_DWI_flair(brain_data, roi_data, tissue_data, age, age_threshold, meta_table, filename, modality_key, downsample=False):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max() - brain.min() + 1e-10)
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()

    syn_parameter, original_parmeter = find_matching_file(age, meta_table, age_threshold, filename, modality_key)

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        mu = syn_parameter[int(ii) * 2]
        std = syn_parameter[int(ii) * 2 + 1]
        len1 = torch.sum(label == ii).item()
        gen_ima[label == ii] = torch.normal(mu, std, (len1,), device=label.device)

    sigma = 0.8
    boundary_mask = torch.zeros_like(label, dtype=torch.bool)
    boundary_mask[:-1, :, :] |= (label[:-1, :, :] != label[1:, :, :])
    boundary_mask[:, :-1, :] |= (label[:, :-1, :] != label[:, 1:, :])
    boundary_mask[:, :, :-1] |= (label[:, :, :-1] != label[:, :, 1:])
    smoothed_img = gen_ima.clone()
    gauss_kernel = get_gauss(sigma).to(brain.device)
    smoothed_values = F.conv3d(smoothed_img, weight=gauss_kernel[None, None, ...], padding=1)
    smoothed_img[boundary_mask] = smoothed_values[boundary_mask]
    smoothed_img[label == 0] = smoothed_img[label != 0].min()

    sigma = 1.5
    smoothed_img = gaussian_filter(smoothed_img.squeeze().detach().cpu().numpy(), sigma=sigma)
    smoothed_img = torch.from_numpy(smoothed_img[None, None]).cuda()

    gen_ima = transform.RandomBiasField()(smoothed_img)
    if downsample:
        gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    
    return gen_ima.squeeze(),tissue