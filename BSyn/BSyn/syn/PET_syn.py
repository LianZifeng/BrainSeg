import torch
import math
import syn.transform as transform
from scipy.ndimage import gaussian_filter
import random


def find_matching_file(age, meta_table, age_threshold):
    matching_files = meta_table[(meta_table['Modality'] != "T2-brain.nii.gz")]
    selected_row = matching_files.sample(n=1)
    syn_parameter = selected_row.iloc[0][3:].tolist()
    return syn_parameter


def GMM_pet(roi_data, brain_data, age, age_threshold, meta_table, downsample=False):
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    ref_img = torch.from_numpy(brain_data).float().cuda()

    syn_parameter = find_matching_file(age,meta_table,age_threshold)

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        mu = syn_parameter[int(ii) * 2]
        sigma = syn_parameter[int(ii) * 2 + 1]
        if math.isnan(mu) or math.isnan(sigma):
            continue
        len1 = torch.sum(label == int(ii)).item()
        gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    gen_ima = transform.fuse_fft_pet(ref_img, gen_ima.squeeze())[None,None]
    sigma = 1.5 
    gen_ima = gaussian_filter(gen_ima.squeeze().detach().cpu().numpy(), sigma=sigma)
    gen_ima = transform.RandomBiasField()(torch.from_numpy(gen_ima[None, None]).cuda())
    if downsample:
        gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)
    gen_ima[label == 0] = gen_ima[label != 0].min()

    return gen_ima.squeeze()