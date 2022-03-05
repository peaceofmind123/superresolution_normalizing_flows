"""
Image denoising with SRFlow
"""
import sys
import natsort, glob, pickle, torch
from collections import OrderedDict
import numpy as np
import os
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

    def normalize_zs(self,  zs, temperature=0.8):
        """
        Given zs as the latent encoding, normalize them
        """


        z_flats = [z.flatten() for z in zs]
        z_flats = torch.cat(z_flats, dim=0)
        sampled_mean, sampled_std = self.sample_mean_and_std(temperature, len(z_flats))

        empirical_mean, empirical_std = z_flats.mean(), z_flats.std(unbiased=False)

        # broadcasted normalization operation
        z_flats = sampled_std / empirical_std * (z_flats - empirical_mean) + sampled_mean

        zs = list(z_flats.split([80*80*6, 40*40*12, 10*10*192]))
        for i in range(len(zs)):
            if i == 0:
                zs[i] = zs[i].view(1,6,80,80)
            elif i == 1:
                zs[i] = zs[i].view(1,12, 40,40)
            else:
                zs[i] = zs[i].view(1,192,10,10)
        return zs
if __name__ == '__main__':
    imgDenoising = ImageDenoising(conf_path)
    zs = imgDenoising.get_multiple_zs('./data/validation/lr/009757.jpg','./data/validation/hr/009757.jpg')
    normalized_zs = imgDenoising.normalize_zs(zs)
