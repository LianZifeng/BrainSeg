import torch
import torch.nn.functional as F
from torch.linalg import multi_dot
import random
import math


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        output = image.clone()
        for transfom in self.transforms:
            output = transfom(output)

        return output


def split_labels(label, index_all):
    """input torch 5D tensor (N, C, D, H, W)"""
    new_label = torch.empty((label.shape[0], len(index_all),) + label.shape[-3:]).to(label.device)

    for j, idx in enumerate(index_all):
        new_label[0, j, ...] = torch.where(label == idx, 1., 0.)

    return new_label


# Training Augmentations
class RandomSkullStrip():
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, label):
        skull_indexs = [30, 62, 85, 165, 258, 259]

        output = torch.clone(label)
        if random.choice((0, 1)):
            if self.verbose: print("skull strip")
            for idx in skull_indexs:
                output[output == idx] = 0

            if random.choice((0, 1)):
                if self.verbose: print("remove CSF")
                output[output == 24] = 0

        return output


class RandomLRFlip():
    def __init__(self, p=0.5, verbose=False):
        self.p = p
        self.verbose = verbose

    def __call__(self, image):
        """input/output 5D tensor (N, C, D, H, W)"""
        if random.random() > self.p:
            if self.verbose: print("flip image")
            return image.flip(dims=(2,))
        else:
            return image


class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        """input/output tensor num_dims >= 3"""
        size = (self.size,) * 3 if type(self.size) == int else self.size

        i = random.randint(0, image.shape[-3] - size[0])
        j = random.randint(0, image.shape[-2] - size[1])
        k = random.randint(0, image.shape[-1] - size[2])

        return image[..., i:i + size[0], j:j + size[1], k:k + size[2]]


class LinearDeform():
    """input/output pytorch 5D tensor (N, C, D, H, W)"""

    def __init__(self, scales=(0.8, 1.2), degrees=(-20, 20), shears=(-0.015, 0.015), trans=(-30, 30), verbose=False):
        self.scales = scales if len(scales) == 6 else scales * 3
        self.degrees = degrees if len(degrees) == 6 else degrees * 3
        self.shears = shears if len(shears) == 6 else shears * 3
        self.trans = trans if len(trans) == 6 else trans * 3
        self.verbose = verbose

    def __call__(self, label):
        max_dim_length = max(label.shape)
        pd = (max(label.shape) - label.shape[-3])
        ph = (max(label.shape) - label.shape[-2])
        pw = (max(label.shape) - label.shape[-1])
        label = F.pad(label, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))  # pad image to cube

        scale_m = torch.diag(1 / torch.tensor([
            random.uniform(self.scales[0], self.scales[1]),
            random.uniform(self.scales[2], self.scales[3]),
            random.uniform(self.scales[4], self.scales[5]),
        ]))  # scale matrix

        angleX = math.radians(random.uniform(self.degrees[0], self.degrees[1]))
        angleY = math.radians(random.uniform(self.degrees[2], self.degrees[3]))
        angleZ = math.radians(random.uniform(self.degrees[4], self.degrees[5]))

        from math import sin, cos
        rotX = torch.tensor([[cos(angleX), 0, sin(angleX)],
                             [0., 1., 0.],
                             [-sin(angleX), 0., cos(angleX)]])

        rotY = torch.tensor([[cos(angleY), -sin(angleY), 0.],
                             [sin(angleY), cos(angleY), 0.],
                             [0., 0., 1.], ])

        rotZ = torch.tensor([[1., 0., 0.],
                             [0., cos(angleZ), -sin(angleZ)],
                             [0., sin(angleZ), cos(angleZ)]])

        shX = random.uniform(self.shears[0], self.shears[1])
        shY = random.uniform(self.shears[2], self.shears[3])
        shZ = random.uniform(self.shears[4], self.shears[5])

        shearX = torch.tensor([
            [1., 0., 0.],
            [shX, 1., 0.],
            [shX, 0., 1.]])
        shearY = torch.tensor([
            [1., shY, 0.],
            [0., 1., 0.],
            [0., shY, 1.]])
        shearZ = torch.tensor([
            [1., 0., shZ],
            [0., 1., shZ],
            [0., 0., 1.]])

        trans = -2 * torch.tensor([
            random.uniform(self.trans[0], self.trans[1]),
            random.uniform(self.trans[2], self.trans[3]),
            random.uniform(self.trans[4], self.trans[5]),
        ]) / 256

        affine = torch.zeros((3, 4))
        affine[:3, :3] = multi_dot((scale_m, rotX, rotY, rotZ, shearX, shearY, shearZ))
        affine[:, 3] = trans

        grid = F.affine_grid(affine[None, ...], label.size(), align_corners=False).to(label.device)
        new_label = F.grid_sample(label, grid, mode='nearest', align_corners=False)

        if self.verbose:
            print(scale_m)
            print(trans)
            print(rotX)
            print(rotY)
            print(rotZ)
            print(shearX)
            print(shearY)
            print(shearZ)
            print(affine, affine.dtype)

        return new_label[..., pd // 2:max_dim_length - (pd - pd // 2), ph // 2:max_dim_length - (ph - ph // 2),
               pw // 2:max_dim_length - (pw - pw // 2)]


class NonlinearDeform():
    def __init__(self, max_std=4):
        self.max_std = max_std

    def __call__(self, label):
        max_dim_length = max(label.shape)
        pd = (max(label.shape) - label.shape[-3])
        ph = (max(label.shape) - label.shape[-2])
        pw = (max(label.shape) - label.shape[-1])
        label = F.pad(label, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))

        # make 3D coordinate grid
        x, y, z = torch.meshgrid(
            torch.linspace(-1, 1, steps=label.shape[-3]),
            torch.linspace(-1, 1, steps=label.shape[-2]),
            torch.linspace(-1, 1, steps=label.shape[-1]),
            indexing="ij")

        std_svf = random.uniform(0, self.max_std)  # sample standard deviation
        svf = torch.normal(0, std_svf, (3, 10, 10, 10),
                           device=label.device)  # 3x10x10x10 simple vector field sampled from Gaussian Distribution
        svf = F.interpolate(svf[None, ...], size=label.shape[-3:], mode="trilinear")[0, ...]  # upsample to image size
        svf = svf / 256 * 2  # rescale
        svf = svf.permute((1, 2, 3, 0))

        grid = torch.stack((x, y, z), dim=-1).float().to(label.device)
        grid = grid - svf
        grid = grid.permute((2, 1, 0, 3))

        deformed_label = F.grid_sample(label, grid[None, ...], mode='nearest', align_corners=False)
        return deformed_label[..., pd // 2:max_dim_length - (pd - pd // 2), ph // 2:max_dim_length - (ph - ph // 2),
               pw // 2:max_dim_length - (pw - pw // 2)]


class GMMSample():
    def __init__(self, mean=(0, 255), std=(0, 35)):
        self.mean = mean
        self.std = std

    def __call__(self, label):
        index_all = torch.unique(label)
        gen_ima = label.clone()
        for ii in index_all:
            mu = random.uniform(self.mean[0], self.mean[1])
            sigma = random.uniform(self.std[0], self.std[1])  # mean and standard deviation
            len1 = torch.sum(label == ii).item()
            gen_ima[label == ii] = torch.normal(mu, sigma, (len1,), device=label.device)

        return gen_ima


class RandomBiasField():
    def __init__(self, max_std=0.6, mode='trilinear'):
        self.max_std = max_std
        self.mode = mode

    def __call__(self, image):
        std = random.uniform(0, self.max_std)
        B = torch.normal(0, std, (4, 4, 4), device=image.device)
        B = F.interpolate(input=B[None, None, ...], size=image.shape[-3:], mode=self.mode)
        B = B[0, 0, ...]
        B = torch.exp(B)
        return image * B


class Rescale():
    def __init__(self, min_max=(0, 1)):
        self.min_max = min_max

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class GammaTransform():
    def __init__(self, std=0.6324):
        self.std = std

    def __call__(self, image):
        return torch.pow(image, torch.exp(torch.normal(0, self.std, image.shape, device=image.device)))


def get_gauss(sigma, kernel_size=3):
    # Calculate the center of the kernel
    center = (kernel_size - 1) / 2

    # Create a 3D coordinate grid
    x = torch.linspace(-center, center, kernel_size)
    y = torch.linspace(-center, center, kernel_size)
    z = torch.linspace(-center, center, kernel_size)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # Calculate the Gaussian function
    gaussian = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that it sums to 1
    gaussian = gaussian / torch.sum(gaussian)

    return gaussian


class RandomDownSample():
    def __init__(self, max_slice_space=5, alpha=(0.98, 1.02), r_hr=1):
        self.r_hr = r_hr
        self.max_slice_space = max_slice_space
        self.alpha = alpha

    def __call__(self, image):
        r_spac = random.uniform(self.r_hr, self.max_slice_space)
        r_thick = random.uniform(self.r_hr, r_spac)
        a = random.uniform(self.alpha[0], self.alpha[1])
        std_thick = (2 * a * math.log(10) * r_thick) / (2 * math.pi * self.r_hr)
        # r_spac = std_thick = 4

        origin_shape = image.size()[-3:]
        sample_shape = (torch.tensor(image.size()[-3:]) / r_spac).int()
        sample_shape = torch.Size(sample_shape.tolist())
        # print(sample_shape)

        gauss_kernel = get_gauss(sigma=std_thick).to(image.device)
        image = F.conv3d(image, weight=gauss_kernel[None, None, ...], padding=1)  # Guassian Blur

        image = F.interpolate(input=image, size=sample_shape, mode='trilinear')  # downsample to low resolution r_spac
        image = F.interpolate(input=image, size=origin_shape, mode='trilinear')  # upsample back to r_hr
        return image


def fuse_fft_mri(ref_in, pre_in):
    ref=ref_in.clone()
    pre=pre_in.clone()
    fft_src_ref = torch.fft.fftn(ref, dim=(-3, -2, -1))
    fshift_ref = torch.fft.fftshift(fft_src_ref).cuda()
    fft_src_pre = torch.fft.fftn(pre, dim=(-3, -2, -1))
    fshift_pre = torch.fft.fftshift(fft_src_pre).cuda()

    a = ref.shape[0] // 2
    b = ref.shape[1] // 2
    c = ref.shape[2] // 2
    s_a = a // 40
    s_b = b // 40
    s_c = c // 40

    fshift_pre[a - s_a:a + s_a, b - s_b:b + s_b, c - s_c:c + s_c] = fshift_ref[a - s_a:a + s_a, b - s_b:b + s_b, c - s_c:c + s_c]

    ishift = torch.fft.ifftshift(fshift_pre).cuda()
    iimg = abs(torch.fft.ifftn(ishift, dim=(-3, -2, -1)).cuda())

    return iimg


def fuse_fft_pet(ref_in, pre_in):
    ref = ref_in.clone()
    pre = pre_in.clone()
    fft_src_ref = torch.fft.fftn(ref, dim=(-3, -2, -1))
    fshift_ref = torch.fft.fftshift(fft_src_ref).cuda()
    fft_src_pre = torch.fft.fftn(pre, dim=(-3, -2, -1))
    fshift_pre = torch.fft.fftshift(fft_src_pre).cuda()

    a = ref.shape[0] // 2
    b = ref.shape[1] // 2
    c = ref.shape[2] // 2
    s_a = a // 10
    s_b = b // 10
    s_c = c // 10

    fshift_pre[a - s_a:a + s_a, b - s_b:b + s_b, c - s_c:c + s_c] = fshift_ref[a - s_a:a + s_a, b - s_b:b + s_b,
                                                                    c - s_c:c + s_c]

    ishift = torch.fft.ifftshift(fshift_pre).cuda()
    iimg = abs(torch.fft.ifftn(ishift, dim=(-3, -2, -1)).cuda())
    mask = (ref_in != 0)
    ref[mask] = iimg[mask]
    return ref