import torch
import math
from scipy.ndimage import gaussian_filter


def find_matching_file(age, meta_table, age_threshold, modality_key):
    matching_files = meta_table[(meta_table['Modality'] == modality_key)]
    selected_row = matching_files.sample(n=1)
    syn_parameter = selected_row.iloc[0][3:].tolist()
    return syn_parameter


def GMM_ct(tissue_data, roi_data, age, age_threshold, meta_table, modality_key, downsample=False):
    label = torch.from_numpy(roi_data[None, None, ...]).float().cuda()
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()

    syn_parameter = find_matching_file(age, meta_table, age_threshold, modality_key)

    index_all = torch.unique(label)
    gen_ima = label.clone()
    for ii in index_all:
        mu = syn_parameter[int(ii) * 2]
        sigma = syn_parameter[int(ii) * 2 + 1]
        if math.isnan(mu) or math.isnan(sigma):
            continue
        len1 = torch.sum(label == int(ii)).item()
        gen_ima[label == int(ii)] = torch.normal(mu, sigma, (len1,), device=label.device)

    sigma = 1.5
    gen_ima = gaussian_filter(gen_ima.squeeze().detach().cpu().numpy(), sigma=sigma)
    gen_ima = torch.from_numpy(gen_ima[None, None]).cuda()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gen_ima[label == 0] = gen_ima[label != 0].min()
    return gen_ima.squeeze(), tissue