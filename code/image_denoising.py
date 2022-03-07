"""
Image denoising with SRFlow
"""
import sys
import natsort, glob, pickle, torch
from collections import OrderedDict
import numpy as np
import os
import copy
import options.options as option
from models import create_model
from imresize import imresize
from test import fiFindByWildcard, imread
from utils.util import opt_get
from PIL import Image
import Measure
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from result_analysis import find_files, pickleRead, t, rgb
from best_outa_n import load_model
import math
conf_path = './confs/SRFlow_CelebA_8X.yml'

class ImageDenoising:
    def __init__(self, conf_path):

        # load the model
        self.model, self.opt = load_model(conf_path)
        a = 1

    def get_multiple_zs(self, lq_img_path, gt_img_path):
        # read lq img
        lq = t(imresize(imread(lq_img_path), output_shape=(20, 20)))
        # read gt img
        gt = t(imresize(imread(gt_img_path), output_shape=(160,160)))

        # providing the epses param will record the intermediate latent codes as well
        return self.model.get_encode_z(lq=lq, gt=gt, epses=[])


    def sample_mean_and_std(self, temperature, num_zs):
        """
        Sample empirical mean and variance from the Normal and gamma distribution respectively
        """
        mu_hat = np.random.normal(loc=0.0, scale=temperature) # the empirical mean
        # the shape (k) and scale (theta) parameters
        k = (num_zs - 1)/2
        theta = (2*temperature)/(num_zs - 1)

        # sample variance from the gamma distribution
        var = np.random.gamma(shape=k, scale=theta)
        sigma_hat = np.sqrt(var)

        return mu_hat, sigma_hat

    def normalize_flattened_zs(self,z_flats, temperature=0.8):
        sampled_mean, sampled_std = self.sample_mean_and_std(temperature, len(z_flats))

        empirical_mean, empirical_std = z_flats.mean(), z_flats.std(unbiased=False)

        # broadcasted normalization operation
        z_flats = sampled_std / empirical_std * (z_flats - empirical_mean) + sampled_mean

        return z_flats

    def global_normalize_zs(self, zs, temperature=0.8):
        """
        Given zs as the latent encoding, perform global normalization on them
        """
        z_flats = [z.flatten() for z in zs]
        z_flats = torch.cat(z_flats, dim=0)
        z_flats = self.normalize_flattened_zs(z_flats,temperature)
        zs = list(z_flats.split([80*80*6, 40*40*12, 10*10*192]))
        for i in range(len(zs)):
            if i == 0:
                zs[i] = zs[i].view(1,6,80,80)
            elif i == 1:
                zs[i] = zs[i].view(1,12, 40,40)
            else:
                zs[i] = zs[i].view(1,192,10,10)
        return zs

    def spatial_normalize_zs(self, zs, temperature=0.8):
        zs_temp = copy.deepcopy(zs)
        for level in range(len(zs_temp)):
            z = zs_temp[level].squeeze(0) # the z we have has leading batch dimension
            for channel in range(len(z)):
                z_level_channel = z[channel]
                z_flats = torch.flatten(z_level_channel)
                z_flats = self.normalize_flattened_zs(z_flats, temperature) # the normalization
                z[channel] = z_flats.view(z_level_channel.shape)

        return zs_temp

    def local_normalize_zs(self, zs, temperature=0.8):
        zs_temp = copy.deepcopy(zs)
        for level in range(len(zs_temp)):
            z = zs_temp[level].squeeze(0)
            c,h,w = z.shape
            # normalization for each level and each pixel position independently
            for i in range(h):
                for j in range(w):
                    z_local = z[:,i,j]
                    z_flats = torch.flatten(z_local)
                    z_flats = self.normalize_flattened_zs(z_flats, temperature) # the local normalization
                    z[:,i,j] = z_flats.view(z_local.shape)

        return zs_temp

    def restore_img(self, degraded_img_path, lq_img_path,temperature=0.8):
        zs = self.get_multiple_zs('./data/validation/hr/009757.jpg', './data/validation/hr/009757.jpg')
        spatial_normalized_zs = self.spatial_normalize_zs(zs, 0.8)
        local_normalized_zs = self.local_normalize_zs(spatial_normalized_zs, 0.8)


if __name__ == '__main__':
    imgDenoising = ImageDenoising(conf_path)
    zs = imgDenoising.get_multiple_zs('./data/validation/lr/009757.jpg','./data/validation/hr/009757.jpg')
    spatial_normalized_zs = imgDenoising.spatial_normalize_zs(zs,0.8)
    local_normalized_zs = imgDenoising.local_normalize_zs(spatial_normalized_zs, 0.8)
    pass
