import torch
import random
import transform as transform
from transform import get_gauss
from scipy.stats import norm
import torch.nn.functional as F


def find_matching_file(age, meta_table, age_threshold, filename, modality_key):
    if age is None:
        print("Warning: Age is None, using default age = 20")
        age = 20

    matching_files = meta_table[(meta_table['Modality'] == modality_key) & (meta_table['Age'].between(age - age_threshold, age + age_threshold))]
    selected_row = matching_files.sample(n=1)
    syn_parameter= selected_row.iloc[0][3:].tolist()

    original_file = meta_table[(meta_table['Modality'] == "brain.nii.gz") & (meta_table['combined_col']==filename)]
    original_parameter = original_file.iloc[0][3:].tolist()

    return syn_parameter, original_parameter


def GMM_mri_t1(brain_data, roi_data, tissue_data, age, age_threshold, meta_table,filename, modality_key):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max()-brain.min())
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()

    syn_parameter,original_parmeter = find_matching_file(age, meta_table, age_threshold, filename, modality_key)

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
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255

    return gen_ima.squeeze()


def GMM_mri_t2(brain_data,roi_data,tissue_data, age, age_threshold, meta_table,filename,modality_key):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max() - brain.min())
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()

    syn_parameter,original_parmeter = find_matching_file(age, meta_table, age_threshold, filename, modality_key)

    index_all = torch.unique(tissue)
    gen_ima = tissue.clone()
    for ii in index_all:
        if ii != 0:
            mu1 = syn_parameter[int(ii) * 2]
            std1 = syn_parameter[int(ii) * 2 + 1]
            mu2 = original_parmeter[int(ii) * 2]
            std2 = original_parmeter[int(ii) * 2 + 1]

            struct_copy = brain.clone()*-1
            # intensity_T2 = struct_copy[tissue == int(ii)]
            # mu2, std2 = norm.fit(intensity_T2)
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
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255
    return gen_ima.squeeze()


def GMM_DWI_flair(brain_data, roi_data, tissue_data, age, age_threshold, meta_table, filename, modality_key):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max() - brain.min())
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()

    syn_parameter,original_parmeter = find_matching_file(age,meta_table,age_threshold,filename,modality_key)

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        if ii != 0:
            mu1 = syn_parameter[int(ii) * 2]
            std1 = syn_parameter[int(ii) * 2 + 1]
            mu2 = original_parmeter[int(ii) * 2]
            std2 = original_parmeter[int(ii) * 2 + 1]

            struct_copy = brain.clone()*-1
            struct_copy[label != int(ii)] = 0
            adjusted_img = (struct_copy - mu2) / std2 * std1 + mu1
            gen_ima[label == int(ii)] = adjusted_img[label == int(ii)]

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

    gen_ima = transform.RandomBiasField()(smoothed_img)
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255
    return gen_ima.squeeze(),tissue