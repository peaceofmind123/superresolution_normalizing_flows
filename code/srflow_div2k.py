"""
Experimentation with the div2k dataset
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
conf_path = './confs/SRFlow_DF2K_4X.yml'
class Div2kModelAnalysis:

    def __init__(self, conf_path):
        # load model
        self.model, self.opt = load_model(conf_path)


    def getSRandSave(self, lq_img_path, sr_save_path, temperature=0.8):
        # read the image
        lq = imresize(imread(lq_img_path), output_shape=(160,160))
        # Sample a super-resolution for a low-resolution image
        sr = rgb(self.model.get_sr(lq=t(lq), heat=temperature))
        plt.imsave(sr_save_path,sr)

    @staticmethod
    def upscaleBicubic(lq_img_path, upscaled_save_path, target_resolution=(640,640)):
        """
        Used for visual comparison of generated sr image and
        the upscaled version of the lr image
        """
        upscaled = imresize(imread(lq_img_path), output_shape=target_resolution)
        plt.imsave(upscaled_save_path,upscaled)

    @staticmethod
    def find_rows_and_cols(initial_resolution):
        """
        Let h,w = initial_resolution
        Find values r, k such that 160*r<= h <= 160*(r+1), similarly for k
        """
        h,w = initial_resolution
        i = 1
        r = 1 # number of rows
        k = 1 # number of cols
        flag = 0
        while flag < 2:
            if 160*i <= h < 160*(i + 1):
                r = i
                flag += 1

            if 160*i <= w < 160*(i + 1):
                k = i
                flag += 1

            i += 1
        return r,k

    @staticmethod
    def find_final_resolution_and_iterations(target_resolution, rows, cols):
        """
        Find the final resolution of the form (160*4^n*rows, 160*4^n*cols)
        from the target resolution
        Also find the required number of iterations
        """
        h,w = target_resolution
        i = 1
        proposed_h, proposed_w = 160*4, 160*4
        flag = 0
        while flag < 2:
            if 160*(4**i)*rows < h <= 160*(4**(i+1))*rows:
                proposed_h = 160*(4**(i+1))*rows # take the upper limit
                flag += 1
            if 160 * (4 ** i) * cols < w <= 160 * (4 ** (i + 1)) * cols:
                proposed_w = 160 * (4 ** (i + 1)) * cols
                flag += 1

            i+=1

        return proposed_h, proposed_w, i

    @staticmethod
    def findNumIterationsHeirarchical(rows, cols, final_resolution):
        """
        Find the number of iterations (heirarchies) to run the upscaling for
        given rows (r) and cols(k)
        Given as: h' = 160*(4^n)*r, w' = 160*(4^n)*k => n = log_4(h'/160r)
        The final resolution should also be adjusted beforehand
        """
        h_, w_ = final_resolution
        # Assuming rows and cols are consistently found
        frac1, frac2 = h_/(160*rows), w_/(160*cols)
        assert f'{frac1:0.2f}' == f'{frac2:0.2f}'

        return int(math.log(frac1,4))


    def heirarchicalUpscaling(self, lq_img_path, num_iterations, output_img_path, temperature=0.8):
        """
        Perform successive cropping and upscaling of image patches of 160x160 from the target image
        """
        lq = imread(lq_img_path)

        initial_resolution = lq.shape[0], lq.shape[1]
        rows,cols = Div2kModelAnalysis.find_rows_and_cols(initial_resolution)

        # get the resolution that is multiple of (160, 160) and the corresponding lq img
        modified_resolution = rows*160, cols*160
        lq = imresize(lq, output_shape=modified_resolution)

        for i in range(num_iterations):
            sr_img = np.empty((rows*(4**(i+1))*160, 160*cols*(4**(i+1)), 3))

            for r in range(rows):
                start_idx_row = 160 * (4**i) * r
                end_idx_row = 160 * (4**i) * (r + 1)
                start_idx_row_sr = 160 * (4**(i+1)) * r
                end_idx_row_sr = 160 * (4 ** (i+1)) * (r + 1)

                for k in tqdm(range(cols), total=cols):
                    start_idx_col = 160 * (4**i) * k
                    start_idx_col_sr = 160 * (4**(i+1)) * k
                    end_idx_col = 160 * (4**i) * (k+1)
                    end_idx_col_sr = 160 * (4**(i+1)) * (k+1)

                    # get the patch to upscale
                    patch = lq[start_idx_row: end_idx_row, start_idx_col: end_idx_col, :]


                    # upscale the image (4x)
                    sr_img[start_idx_row_sr: end_idx_row_sr, start_idx_col_sr: end_idx_col_sr] = rgb(self.model.get_sr(lq=t(patch), heat=temperature))

            lq = sr_img

        plt.imsave(output_img_path, lq)

def generateSRImageTest():
    div2kAnalysis = Div2kModelAnalysis(conf_path)
    div2kAnalysis.getSRandSave('./data/div2k/test/lr/remote_sensing/Data2.png','./data/div2k/test/sr/remote_sensing/Data2.png')

def findRowsAndColsTest():
    assert Div2kModelAnalysis.find_rows_and_cols((1080, 1920)) == (6,12)

def find_final_res_iters_test():
    print(Div2kModelAnalysis.find_final_resolution_and_iterations((1300,2200),6,12))

def heirarchical_upscaling_test():
    div2kAnalysis = Div2kModelAnalysis(conf_path)
    div2kAnalysis.heirarchicalUpscaling('./data/div2k/test/sr/remote_sensing/Data2.png',
                                        1,
                                        './data/div2k/test/sr/remote_sensing/Data2_heir.png')
if __name__ == '__main__':
    #Div2kModelAnalysis.upscaleBicubic('./data/div2k/test/lr/remote_sensing/10mc.jpg',
     #                                 './data/div2k/test/upscaled_bicubic/remote_sensing/10mc.png')
    #generateSRImageTest()
    #findRowsAndColsTest()
    #find_final_res_iters_test()
    heirarchical_upscaling_test()
