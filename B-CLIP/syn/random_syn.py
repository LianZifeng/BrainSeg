import torch
import random
import syn.transform as transform


def GMM_random(tissue_data):
    tissue = torch.from_numpy(tissue_data[None, None, ...]).float().cuda()

    mean,std = (0, 255), (0, 35)
    gen_ima = tissue * 0

    index_all = torch.unique(tissue)
    for ii in index_all:

        mu = random.uniform(mean[0], mean[1])
        sigma = random.uniform(std[0], std[1])
        len1 = torch.sum(tissue == ii).item()
        gen_ima[tissue == ii] = torch.normal(mu, sigma, (len1,), device=tissue.device)

    gen_ima = transform.RandomBiasField()(gen_ima)
    gen_ima = transform.RandomDownSample()(gen_ima)
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min())
    gamma=random.uniform(0.8, 1.2)
    gen_ima = torch.pow(gen_ima, gamma)
    gen_ima[tissue == 0] = gen_ima[tissue != 0].min()
    gen_ima = (gen_ima - gen_ima.min()) / (gen_ima.max() - gen_ima.min()) * 255
    return gen_ima.squeeze()