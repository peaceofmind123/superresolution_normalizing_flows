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
from image_denoising import Noise, ImageDenoising

conf_path = './confs/SRFlow_CelebA_8X.yml'
dataroot_lr = './data/validation/lr'
dataroot_gt = './data/validation/hr'

df_avgs_before_save_path = './data/validation/denoising/avgs-before.csv'
df_avgs_after_save_path = './data/validation/denoising/avgs-after.csv'

class DenoisingAnalysis:

    def __init__(self,conf_path, dataroot_lr, dataroot_gt):
        # init image denoising object
        self.denoiser = ImageDenoising(conf_path)

        # load the data
        self.load_data(dataroot_lr,dataroot_gt)

        # instantiate measure
        self.measure = Measure.Measure()

        # initialize the temperatures to sample at
        self.temperatures = [x for x in np.linspace(0, 1, num=11)]
        self.filenames = [p.split('/')[-1] for p in self.lq_paths]

    def load_data(self, dataroot_lr, dataroot_gt):
        # find filenames inside root folder
        self.lq_paths = fiFindByWildcard(os.path.join(dataroot_lr, '*.jpg'))
        self.gt_paths = fiFindByWildcard(os.path.join(dataroot_gt, '*.jpg'))
        self.lq_paths.sort()
        self.gt_paths.sort()

        # verify if lq_paths filenames and gt_paths filenames agree on all indices
        for i in range(len(self.lq_paths)):
            assert self.lq_paths[i].split('/')[-1] == self.gt_paths[i].split('/')[-1]

        # read the data in
        self.lqs = [imread(p) for p in self.lq_paths]
        self.gts = [imread(p) for p in self.gt_paths]

    def initialize_dataframe(self):
        """
        Initialize the dataframe that will hold the averages of measures before and after
        denoising at various temperatures
        """
        temperatures = [x for x in np.linspace(0, 1, num=11)]
        self.avgs_after = pd.DataFrame(columns=['psnr', 'ssim', 'lpips'],
                          index=temperatures)

        # this df will have only one row
        self.avgs_before = pd.DataFrame(columns=['psnr', 'ssim', 'lpips'], index=[0])

    def denoising_analysis(self,  noise:Noise, df_avg_before_save_path, df_avgs_after_save_path):
        """
        Perform denoising analysis for the particular noise type
        The noise object has to be constructed beforehand and passed in
        """
        # initialize dataframe to hold the average values of the measures
        self.initialize_dataframe()

        # initialize sums before and after denoising
        sum_psnr_before = 0.0
        sum_ssim_before = 0.0
        sum_lpips_before = 0.0

        sums_psnr_after = np.array([0.0 for _ in np.linspace(0,1,num=11)])
        sums_ssim_after = copy.deepcopy(sums_psnr_after)
        sums_lpips_after = copy.deepcopy(sums_psnr_after)

        for lq, gt, fname in tqdm(zip(self.lqs, self.gts, self.filenames), total=len(self.lqs)):
            # add noise
            noisy_gt = noise.add_noise(gt)
            # calculate psnr, ssim and lpips before denoising
            psnr, ssim, lpips = self.measure.measure(cast_to_uint8(noisy_gt), gt)
            sum_psnr_before += psnr
            sum_ssim_before += ssim
            sum_lpips_before += lpips

            # perform denoising at various temperatures
            for i, temperature in enumerate(np.linspace(0, 1, num=11)):
                restored_img = self.denoiser.restore_degraded_img(noisy_gt, temperature)
                # calculate psnr, ssim and lpips after denoising
                psnr, ssim, lpips = self.measure.measure(restored_img, gt)
                sums_psnr_after[i] += psnr
                sums_ssim_after[i] += ssim
                sums_lpips_after[i] += lpips


        # convert sums into averages
        num = len(self.lqs)
        sum_psnr_before /= num
        sum_ssim_before /= num
        sum_lpips_before /= num
        sums_psnr_after /= num
        sums_ssim_after /= num
        sums_lpips_after /= num

        # fill in the dataframes
        self.avgs_before['psnr'][0] = sum_psnr_before
        self.avgs_before['ssim'][0] = sum_ssim_before
        self.avgs_before['lpips'][0] = sum_lpips_before

        self.avgs_after['psnr'] = sums_psnr_after
        self.avgs_after['ssim'] = sums_ssim_after
        self.avgs_after['lpips'] = sums_lpips_after

        self.avgs_before.to_csv(df_avg_before_save_path)
        self.avgs_after.to_csv(df_avgs_after_save_path)



def denoisingClassInitTest():
    """Seems like it's passing"""
    denoisingAnalysis = DenoisingAnalysis(conf_path,dataroot_lr, dataroot_gt)
    pass


def initDataframesTest():
    """Seems like it's passing"""
    denoisingAnalysis = DenoisingAnalysis(conf_path, dataroot_lr, dataroot_gt)
    denoisingAnalysis.initialize_dataframe()

def fillDfsTest():
    """Test passing"""
    denoisingAnalysis = DenoisingAnalysis(conf_path, dataroot_lr, dataroot_gt)
    denoisingAnalysis.initialize_dataframe()
    sum_psnr_before, sum_ssim_before, sum_lpips_before = np.random.randn(3)
    sums_psnrs_after, sums_ssims_after, sums_lpips_after = [np.random.randn(11) for _ in range(3)]
    avgs_before = denoisingAnalysis.avgs_before
    avgs_after = denoisingAnalysis.avgs_after

    avgs_before['psnr'][0] = sum_psnr_before
    avgs_before['ssim'][0] = sum_ssim_before
    avgs_before['lpips'][0] = sum_lpips_before

    avgs_after['psnr'] = sums_psnrs_after
    avgs_after['ssim'] = sums_ssims_after
    avgs_after['lpips'] = sums_lpips_after

    avgs_before.to_csv('./data/validation/denoising/test-avgs-before.csv')
    avgs_after.to_csv('./data/validation/denoising/test-avgs-after.csv')

def runDenoisingAnalysis():
    denoisingAnalysis = DenoisingAnalysis(conf_path, dataroot_lr, dataroot_gt)
    # on gaussian noise model
    noise = Noise('gaussian', mean=0.0, std=20.0)
    denoisingAnalysis.denoising_analysis(noise,df_avgs_before_save_path,df_avgs_after_save_path)


# helper method
def cast_to_uint8(img):
    min_, max_ = img.min(), img.max()
    img = (img - min_) / (max_ - min_) * 255
    return img.astype(np.uint8)

if __name__ == '__main__':
    runDenoisingAnalysis()

