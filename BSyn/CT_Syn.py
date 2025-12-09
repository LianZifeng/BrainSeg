import torch
import random
import transform as transform
import math


def find_matching_file(age, meta_table, age_threshold, modality_key):
    if age is None:
        print("Warning: Age is None, using default age = 20")
        age = 20
    matching_files = meta_table[(meta_table['Modality'] == modality_key) & (meta_table['Age'].between(age - age_threshold, age + age_threshold))]

    selected_row = matching_files.sample(n=1)
    syn_parameter = selected_row.iloc[0][3:].tolist()

    return syn_parameter


def GMM_ct(tissue_data, roi_data, age, age_threshold, meta_table, modality_key):
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()
    syn_parameter = find_matching_file(age,meta_table,age_threshold, modality_key)

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        mu = syn_parameter[int(ii) * 2]
        sigma = syn_parameter[int(ii) * 2 + 1]
        if math.isnan(mu) or math.isnan(sigma):
            continue
        len1 = torch.sum(label == int(ii)).item()
        gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    gen_ima = transform.RandomBiasField()(gen_ima)
    gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)

    gen_ima[label == 0] = gen_ima[label != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255

    return gen_ima.squeeze(),tissue