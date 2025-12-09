import torch
import numpy as np
import transform as transform
import os
import math
import SimpleITK as sitk
from scipy.ndimage import affine_transform
import random
from scipy.ndimage import zoom
from scipy.ndimage import label, find_objects


def find_matching_file(age, meta_table, age_threshold, modality_key):
    if age is None:
        print("Warning: Age is None, using default age = 20")
        age = 20
    matching_files = meta_table[(meta_table['Modality'] == modality_key) & (meta_table['Age'].between(age - age_threshold, age + age_threshold))]
    selected_row = matching_files.sample(n=1)
    syn_parameter = selected_row.iloc[0][3:].tolist()
    return syn_parameter


def find_matching_file_mri(age, meta_table, age_threshold, filename, modality_key):
    matching_files = meta_table[(meta_table['Modality'] == modality_key) & (meta_table['Age'].between(age - age_threshold, age + age_threshold))]
    selected_row = matching_files.sample(n=1)
    syn_parameter = selected_row.iloc[0][3:].tolist()
    original_file = meta_table[(meta_table['Modality'] == "brain.nii.gz") & (meta_table['combined_col'] == filename)]
    original_parameter = original_file.iloc[0][3:].tolist()
    return syn_parameter,original_parameter


def random_transform_centered(tumor_data):
    shape = tumor_data.shape
    if any(150 > dim > 100 for dim in tumor_data.shape):
        new_tumor_data = tumor_data
        new_shape = shape
        center = (np.array(shape) / 2).astype(int)
        scale = random.uniform(0.5, 1)
    elif any(dim > 150 for dim in tumor_data.shape):
        max_dim = max(tumor_data.shape)
        if max_dim > 150:
            scale_factor = 150 / max_dim
        else:
            scale_factor = 1

        new_tumor_data = zoom(tumor_data, scale_factor, order=0)
        new_shape = new_tumor_data.shape
        center = (np.array(new_shape) / 2).astype(int)
        scale = random.uniform(0.5, 1)
    else:
        new_shape = (np.array(shape) * 1.5).astype(int)
        center = (np.array(new_shape) / 2).astype(int)
        new_tumor_data = np.zeros((new_shape[0], new_shape[1], new_shape[2]))
        new_tumor_data[center[0] - shape[0] // 2:center[0] - shape[0] // 2 + shape[0],
        center[1] - shape[1] // 2:center[1] - shape[1] // 2 + shape[1],
        center[2] - shape[2] // 2:center[2] - shape[2] // 2 + shape[2]] = tumor_data
        scale = random.uniform(0.5, 1.5)

    angle = random.uniform(0, 360)  # 随机旋转角度

    # scale = 1.2

    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))
    rotation_matrix = np.array([
        [scale * cos_theta, -scale * sin_theta, 0],
        [scale * sin_theta, scale * cos_theta, 0],
        [0, 0, 1]
    ])

    full_affine = np.eye(4)
    full_affine[:3, :3] = rotation_matrix

    offset = center - rotation_matrix @ center

    transformed_tumor = affine_transform(
        new_tumor_data,
        rotation_matrix,
        offset=offset,
        order=0,
        mode='constant',
        cval=0
    )

    return transformed_tumor, center, new_shape


def place_tumor(label, tissue, tumor_data, center, new_shape):
    label_copy = label * 0
    tissue_copy = tissue * 0
    label_tumor = label.copy()
    tissue_tumor = tissue.copy()

    non_zero_coords = np.nonzero(label != 0)
    random_idx = random.choice(range(non_zero_coords[0].shape[0]))
    random_pos = [non_zero_coords[0][random_idx],non_zero_coords[1][random_idx], non_zero_coords[2][random_idx]]
    start_end_list = []
    if random_pos[0] - center[0] < 0:
        start_end_list.append((0, new_shape[0]))
    elif random_pos[0] - center[0] + new_shape[0] > label.shape[0]:
        start_end_list.append((label.shape[0] - new_shape[0], label.shape[0]))
    else:
        start_end_list.append((random_pos[0] - center[0],random_pos[0] - center[0] + new_shape[0]))

    if random_pos[1] - center[1] < 0:
        start_end_list.append((0, new_shape[1]))
    elif random_pos[1] - center[1] + new_shape[1] > label.shape[1]:
        start_end_list.append((label.shape[1]-new_shape[1], label.shape[1]))
    else:
        start_end_list.append((random_pos[1]-center[1], random_pos[1] - center[1] + new_shape[1]))

    if random_pos[2] - center[2] < 0:
        start_end_list.append((0, new_shape[2]))
    elif random_pos[2] - center[2] + new_shape[2] > label.shape[2]:
        start_end_list.append((label.shape[2] - new_shape[2], label.shape[2]))
    else:
        start_end_list.append((random_pos[2] - center[2], random_pos[2] - center[2] + new_shape[2]))

    label_copy[start_end_list[0][0]:start_end_list[0][1], start_end_list[1][0]:start_end_list[1][1], start_end_list[2][0]:start_end_list[2][1]] = tumor_data
    tissue_copy[start_end_list[0][0]:start_end_list[0][1], start_end_list[1][0]:start_end_list[1][1], start_end_list[2][0]:start_end_list[2][1]] = tumor_data
    label[label_copy == 1] = 107
    label[label_copy == 2] = 108
    label[label_copy == 3] = 109
    label[label_copy == 4] = 110
    label[label_tumor == 0] = 0

    tissue[tissue_copy == 1] = 107
    tissue[tissue_copy == 2] = 108
    tissue[tissue_copy == 3] = 109
    tissue[tissue_copy == 4] = 110
    tissue[tissue_tumor == 0] = 0

    return label,tissue


def find_bounding_box(image):
    bbox_slices = find_objects(image.astype(np.int32))
    bbox_coordinates = [(bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop, bbox[2].start, bbox[2].stop) for bbox in bbox_slices]
    bb_image = image[bbox_coordinates[0][0]:bbox_coordinates[0][1], bbox_coordinates[0][2]:bbox_coordinates[0][3], bbox_coordinates[0][4]:bbox_coordinates[0][5]]
    return bb_image


def GMM_ct_lesion(roi_data, tissue_data, age, age_threshold, meta_table, meta_tumor, modality_key, lesion_path):
    syn_parameter = find_matching_file(age, meta_table, age_threshold, modality_key)
    matching_tumor = meta_tumor[(meta_tumor['Modality'] == modality_key)]
    selected_row = matching_tumor.sample(n=1)
    tumor_syn_parameter = selected_row.iloc[0][3:].tolist()
    tumor_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lesion_path,selected_row.iloc[0][0], "lesion.nii.gz")))
    tumor_data,center,new_shape = random_transform_centered(tumor_data)
    tumor = torch.from_numpy(tumor_data[None, None, ...]).float().cuda()
    gen_tumor = tumor.clone()
    gen_tumor = transform.LinearDeform()(gen_tumor)
    roi_data, tissue_data = place_tumor(roi_data,tissue_data,gen_tumor.squeeze().detach().cpu().numpy(),center,new_shape)
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        if ii != 0 and ii < 107:
            mu = syn_parameter[int(ii) * 2]
            sigma = syn_parameter[int(ii) * 2 + 1]
            if math.isnan(mu) or math.isnan(sigma):
                continue
            len1 = torch.sum(label == int(ii)).item()
            gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)
        elif int(ii) >= 107:
            random_stat = random.randint(0, 1)  # 随机旋转角度
            if random_stat == 0:
                mu = tumor_syn_parameter[(int(ii) - 106) * 2]
                sigma = tumor_syn_parameter[(int(ii) - 106) * 2 + 1]
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)
            else:
                mu = random.uniform(0, 255)
                sigma = random.uniform(0, 35)
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    gen_ima = transform.RandomBiasField()(gen_ima)
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima.min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255

    label[label >= 107] = 0
    tissue[tissue >= 107] = 0
    return gen_ima.squeeze(), label, tissue


def real_ct_lesion(brain_data, roi_data, tissue_data, age, age_threshold, meta_table, meta_tumor, modality_key, lesion_path):
    matching_tumor = meta_tumor[(meta_tumor['Modality'] == modality_key)]
    selected_row = matching_tumor.sample(n=1)
    tumor_syn_parameter = selected_row.iloc[0][3:].tolist()
    tumor_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lesion_path,selected_row.iloc[0][0], "lesion.nii.gz")))
    tumor_data,center,new_shape = random_transform_centered(tumor_data)
    tumor = torch.from_numpy(tumor_data.astype(np.float32)[None, None, ...]).float().cuda()
    gen_tumor = tumor.clone()
    gen_tumor = transform.LinearDeform()(gen_tumor)
    roi_data, tissue_data = place_tumor(roi_data,tissue_data,gen_tumor.squeeze().detach().cpu().numpy(),center,new_shape)
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max() - brain.min())

    index_all = torch.unique(label)
    gen_ima = brain.clone()
    for ii in index_all:
        if int(ii) >= 107:
            random_stat = random.randint(0, 1)
            if random_stat == 0:
                mu = tumor_syn_parameter[(int(ii) - 106) * 2]
                sigma = tumor_syn_parameter[(int(ii)-106) * 2 + 1]
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)
            else:
                mu = random.uniform(0, 255)
                sigma = random.uniform(0, 35)
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    gen_ima = transform.RandomBiasField()(gen_ima)
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima.min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255

    label[label >= 107] = 0
    tissue[tissue >= 107] = 0
    return gen_ima.squeeze(), label, tissue


def GMM_DWI_flair_lesion(brain_data, roi_data, tissue_data, age, age_threshold, meta_table, filename, meta_tumor, lesion_path, modality_key):
    syn_parameter,original_parmeter = find_matching_file_mri(age,meta_table, age_threshold, filename, modality_key)

    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain - brain.min()) / (brain.max()-brain.min())

    matching_tumor = meta_tumor[(meta_tumor['Modality'] == modality_key)]
    selected_row = matching_tumor.sample(n=1)
    tumor_syn_parameter = selected_row.iloc[0][3:].tolist()
    tumor_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lesion_path,selected_row.iloc[0][0],"lesion.nii.gz")))
    tumor_data,center,new_shape = random_transform_centered(tumor_data)
    tumor = torch.from_numpy(tumor_data[None, None, ...]).float().cuda()
    gen_tumor = tumor.clone()
    gen_tumor = transform.LinearDeform()(gen_tumor)
    roi_data,tissue_data = place_tumor(roi_data,tissue_data,gen_tumor.squeeze().detach().cpu().numpy(),center,new_shape)
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        if ii != 0 and ii < 107:
            mu1 = syn_parameter[int(ii) * 2]
            std1 = syn_parameter[int(ii) * 2 + 1]
            mu2 = original_parmeter[int(ii) * 2]
            std2 = original_parmeter[int(ii) * 2 + 1]

            if ii == 91 or ii == 92:
                struct_copy = brain.clone() * -1
                struct_copy[label != int(ii)] = 0
                adjusted_img = (struct_copy + mu2) / std2 * std1 + mu1
            else:
                struct_copy = brain.clone()
                struct_copy[label != int(ii)] = 0
                adjusted_img = (struct_copy - mu2) / std2 * std1 + mu1
            gen_ima[label == int(ii)] = adjusted_img[label == int(ii)]

        elif int(ii)>=107:
            random_stat = random.randint(0, 1)
            if random_stat == 0:
                mu = tumor_syn_parameter[(int(ii) - 106) * 2]
                sigma = tumor_syn_parameter[(int(ii)-106) * 2 + 1]
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)
            else:
                mu = random.uniform(0, 255)
                sigma = random.uniform(0, 35)
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    gen_ima = transform.RandomBiasField()(gen_ima)
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255

    label[label >= 107] = 0
    tissue[tissue >= 107] = 0
    return gen_ima.squeeze(), label, tissue


def real_DWI_flair_lesion(brain_data, roi_data, tissue_data, meta_tumor, lesion_path, modality_key):
    brain = torch.from_numpy(brain_data[None, None, ...]).float().cuda()
    brain = 255 * (brain-brain.min()) / (brain.max()-brain.min())

    matching_tumor = meta_tumor[(meta_tumor['Modality'] == modality_key)]
    selected_row = matching_tumor.sample(n=1)
    tumor_syn_parameter = selected_row.iloc[0][3:].tolist()
    tumor_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lesion_path,selected_row.iloc[0][0], "lesion.nii.gz")))
    tumor_data,center,new_shape = random_transform_centered(tumor_data)
    tumor = torch.from_numpy(tumor_data.astype(np.float32)[None, None, ...]).float().cuda()
    gen_tumor = tumor.clone()
    gen_tumor = transform.LinearDeform()(gen_tumor)
    roi_data,tissue_data = place_tumor(roi_data,tissue_data,gen_tumor.squeeze().detach().cpu().numpy(),center,new_shape)
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()

    index_all = torch.unique(label)
    gen_ima = brain.clone()
    for ii in index_all:

        if int(ii) >= 107:
            random_stat = random.randint(0, 1)
            if random_stat == 0:
                mu = tumor_syn_parameter[(int(ii) - 106) * 2]
                sigma = tumor_syn_parameter[(int(ii) - 106) * 2 + 1]
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)
            else:
                mu = random.uniform(0, 255)
                sigma = random.uniform(0, 35)
                len1 = torch.sum(label == int(ii)).item()
                gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    gen_ima = transform.RandomBiasField()(gen_ima)
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255

    label[label >= 107] = 0
    tissue[tissue >= 107] = 0
    return gen_ima.squeeze(), label, tissue