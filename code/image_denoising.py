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
from random import choices

conf_path = './confs/SRFlow_CelebA_8X.yml'
plt.show()
import scipy.stats as st

class ImageDenoising:
    def __init__(self, conf_path):

        # load the model
        self.model, self.opt = load_model(conf_path)


    def get_multiple_zs(self, lq_img, gt_img):
        # read lq img
        lq = t(imresize(lq_img, output_shape=(20, 20)))
        # read gt img
        gt = t(imresize(gt_img, output_shape=(160,160)))

        # providing the epses param will record the intermediate latent codes as well
        return self.model.get_encode_z(lq=lq, gt=gt, epses=[])


    def sample_mean_and_std(self, temperature, num_zs):
        """
        Sample empirical mean and variance from the Normal and gamma distribution respectively
        """
        mu_hat = np.random.normal(loc=0.0, scale=temperature / num_zs) # the empirical mean
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
        z_flats = (sampled_std / empirical_std) * (z_flats - empirical_mean) + sampled_mean

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

    def restore_img(self, degraded_img, lq_img,temperature=0.8):
        zs = self.get_multiple_zs(lq_img, gt_img=degraded_img)
        spatial_normalized_zs = self.spatial_normalize_zs(zs, 0.8)
        local_normalized_zs = self.local_normalize_zs(spatial_normalized_zs, 0.8)
        restored = rgb(self.model.get_sr(lq=t(lq_img), heat=temperature,epses=local_normalized_zs))
        return restored

    def add_gaussian_noise(self, gt_img, mean, std):
        noise = np.random.normal(loc=mean, scale=std, size=gt_img.shape).astype(int)

        return gt_img + noise

    def restore_degraded_img(self, degraded_img, temperature=0.8):
        """A helper method"""
        lq_img = imresize(degraded_img, output_shape=(20,20))
        return self.restore_img(degraded_img,lq_img,temperature)


class Noise:
    def __init__(self, type, **kwargs):
        self.type = type
        self.kwargs = kwargs

        if type == 'gaussian':
            self.noise_func = Noise.add_gaussian_noise
        elif type == 'rayleigh':
            self.noise_func = Noise.add_rayleigh_noise
        elif type == 'gamma':
            self.noise_func = Noise.add_gamma_noise
        elif type == 'exp':
            self.noise_func = Noise.add_exp_noise
        elif type == 'salt_pepper':
            self.noise_func = Noise.add_salt_pepper_noise
        elif type == 'uniform':
            self.noise_func = Noise.add_uniform_noise
        else:
            raise NotImplementedError(f'Noise model {type} has not been implemented!')

    def add_noise(self, gt_img):
        """Add noise, assuming correct noise model params are provided in kwargs"""
        return self.noise_func(copy.deepcopy(gt_img), self.kwargs)

    @staticmethod
    def add_gaussian_noise(gt_img, kwargs):
        mean = kwargs['mean']
        std = kwargs['std']

        noise = np.random.normal(loc=mean, scale=std, size=gt_img.shape).astype(int)

        return gt_img + noise

    @staticmethod
    def add_rayleigh_noise(gt_img,kwargs):
        scale = kwargs['scale']
        noise = np.random.rayleigh(scale=scale, size=gt_img.shape).astype(int)
        return gt_img + noise

    @staticmethod
    def add_gamma_noise(gt_img,kwargs):
        shape = kwargs['shape']
        scale = kwargs['scale']

        noise = np.random.gamma(shape=shape, scale=scale, size=gt_img.shape).astype(int)
        # clamping can be done, but it will not be representative of the noise model
        # noisy_img = gt_img + noise
        # noisy_img = np.where(noisy_img < 0,0,noisy_img)
        # noisy_img = np.where(noisy_img > 255, 255, noisy_img)

        return gt_img + noise

    @staticmethod
    def add_exp_noise(gt_img, kwargs):
        scale = kwargs['scale']
        noise = np.random.exponential(scale=scale, size=gt_img.shape).astype(int)
        return gt_img + noise

    @staticmethod
    def add_salt_pepper_noise(gt_img, kwargs):
        salt_prob = kwargs['salt_prob']
        pepper_prob = kwargs['pepper_prob']

        # sample random numbers from uniform distribution from 0-1
        p1 = np.random.uniform(low=0.0, high=1.0,size=gt_img.shape)
        p2 = np.random.uniform(low=0.0, high=1.0, size=gt_img.shape)
        noisy_image = np.where(p1<=salt_prob,0,gt_img)
        noisy_image = np.where(p2<=pepper_prob, 255, noisy_image)
        return noisy_image

    @staticmethod
    def add_uniform_noise(gt_img,kwargs):
        low = kwargs['low']
        high = kwargs['high']
        noise = np.random.uniform(low=low, high=high, size=gt_img.shape).astype(int)
        return gt_img + noise


def image_write(img, path):
    plt.imsave(path,img)

def downsample_and_write(gt_img_path, lq_img_path):
    lq = imresize(imread(gt_img_path), output_shape=(20, 20))
    image_write(lq,lq_img_path)


def downsample_expt():
    gt_img_paths = ['./data/camilia.jpeg', './data/sansa.jpeg', './data/srinkhala.jpeg']
    lq_img_paths = ['./data/camilia-lq.png', './data/sansa-lq.jpeg', './data/srinkhala-lq.jpeg']

    for gt_img_path, lq_img_path in zip(gt_img_paths, lq_img_paths):
        downsample_and_write(gt_img_path,lq_img_path)

def get_sr_with_epses_expt():
    """Test passing"""
    lq_img_path = './data/sansa-lq.jpeg'
    gt_img_path = './data/sansa.jpeg'
    lq = t(imresize(imread(lq_img_path), output_shape=(20, 20)))
    imgDenoising = ImageDenoising(conf_path)
    zs = imgDenoising.get_multiple_zs(lq_img_path, gt_img_path)
    zs = imgDenoising.local_normalize_zs(zs)
    sr = rgb(imgDenoising.model.get_sr(lq=lq, heat=0.8, epses=zs))
    plt.imshow(sr)
    plt.show()

def denoising_with_noise_expt( gt_img_path,noise_type='gaussian',):
    imgDenoising = ImageDenoising(conf_path)

    noise_model = Noise(noise_type,)
    noisy_img = imgDenoising.add_gaussian_noise(gt, mean, std)
    plt.imshow(noisy_img)
    plt.show()
    restored = imgDenoising.restore_degraded_img(noisy_img)
    plt.imshow(restored)
    plt.show()
    pass

if __name__ == '__main__':

    noise = Noise(type='gaussian',mean=0.0, std=20.0)
    gt_img = imread('./data/sansa.jpeg')
    noisy_image = noise.add_noise(gt_img)

    denoising_with_noise_expt('./data/sansa.jpeg')

    # get_sr_with_epses_expt()
    #
    # imgDenoising = ImageDenoising(conf_path)
    #
    # lq_img_path = './data/validation/lr/009757.jpg'
    # gt_img_path = './data/validation/hr/009757.jpg'
    # denoised_img_path = './data/validation/denoising/009757.jpg'
    #
    # denoised_img = imgDenoising.restore_img(degraded_img_path=gt_img_path, lq_img_path=lq_img_path, temperature=0.8)
    # plt.imshow(denoised_img)
    # plt.show()
    # pass
