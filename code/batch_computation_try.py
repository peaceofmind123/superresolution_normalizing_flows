import sys
import natsort, glob, pickle, torch
from collections import OrderedDict
import numpy as np
import os
import options.options as option
from models import create_model
from imresize import imresize
from test import load_model, fiFindByWildcard, imread
from PIL import Image
import Measure
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from result_analysis import find_files,pickleRead,t,rgb

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 22
conf_path = './confs/SRFlow_CelebA_8X.yml'

dataroot_lr = './data/validation/lr'
dataroot_gt = './data/validation/hr'

df_lpips_save_path = './data/validation/df_lpips.csv'
df_ssim_save_path ='./data/validation/df_ssim.csv'
df_psnr_save_path = './data/validation/df_psnr.csv'

avg_psnr_save_path = './data/validation/avg_psnr.pkl'
avg_lpips_save_path = './data/validation/avg_lpips.pkl'
avg_ssim_save_path = './data/validation/avg_ssim.pkl'

df_averages_save_path = './data/validation/df_averages.csv'


