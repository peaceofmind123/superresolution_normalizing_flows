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

    def get_multiple_zs(self, lq, gt):
        pass

if __name__ == '__main__':
    imgDenoising = ImageDenoising(conf_path)

