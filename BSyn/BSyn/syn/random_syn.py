import torch
import random
from scipy.ndimage import gaussian_filter
import syn.transform as transform


def GMM_random(label=None, age=None, age_threshold=None, meta_table=None, downsample=False):
    label = torch.from_numpy(label[None, None, ...]).float().cuda()
    mean,std = (0, 255), (0, 35)
    gen_ima = label * 0

    index_all = torch.unique(label)
    for ii in index_all:
        mu = random.uniform(mean[0], mean[1])
        sigma = random.uniform(std[0], std[1])
        len1 = torch.sum(label == ii).item()
        gen_ima[label == ii] = torch.normal(mu, sigma, (len1,), device=label.device)
        
    sigma = 1.5
    gen_ima = gaussian_filter(gen_ima.squeeze().detach().cpu().numpy(), sigma=sigma)
    gen_ima = torch.from_numpy(gen_ima[None, None])
    gen_ima = transform.RandomBiasField()(gen_ima)
    if downsample:
        gen_ima = transform.RandomDownSample(max_slice_space=5)(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma = random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)
    gen_ima[label == 0] = gen_ima[label != 0].min()

    return gen_ima.squeeze()