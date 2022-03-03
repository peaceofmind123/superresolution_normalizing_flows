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

def generateSRImageTest():
    div2kAnalysis = Div2kModelAnalysis(conf_path)
    div2kAnalysis.getSRandSave('./data/div2k/test/lr/remote_sensing/Data2.png','./data/div2k/test/sr/remote_sensing/Data2.png')

if __name__ == '__main__':
    #Div2kModelAnalysis.upscaleBicubic('./data/div2k/test/lr/remote_sensing/10mc.jpg',
     #                                 './data/div2k/test/upscaled_bicubic/remote_sensing/10mc.png')
    generateSRImageTest()