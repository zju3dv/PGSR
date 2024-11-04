#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def mean_filter(img: torch.Tensor, ksize: int = 5):
    pad = (ksize - 1) // 2
    img = F.pad(img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.avg_pool2d(img, kernel_size=ksize, stride=1, padding=0)
    return out

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img

def normalize_rgb(img):
    r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
    r = normalize(r)
    g = normalize(g)
    b = normalize(b)
    return torch.stack([r, g, b], dim=0)