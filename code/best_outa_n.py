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
from result_analysis import find_files,pickleRead,t,rgb, generateGraphOldvNew

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 22
conf_path = './confs/SRFlow_CelebA_8X.yml'

dataroot_lr = './data/validation/lr'
dataroot_gt = './data/validation/hr'

df_lpips_save_path = './data/validation/df_lpips.csv'
df_ssim_save_path ='./data/validation/df_ssim.csv'
df_psnr_save_path = './data/validation/df_psnr.csv'

df_lpips_save_path_old = './data/validation/df_lpips_old.csv'
df_ssim_save_path_old ='./data/validation/df_ssim_old.csv'
df_psnr_save_path_old = './data/validation/df_psnr_old.csv'

avg_psnr_save_path = './data/validation/avg_psnr.pkl'
avg_lpips_save_path = './data/validation/avg_lpips.pkl'
avg_ssim_save_path = './data/validation/avg_ssim.pkl'

df_averages_save_path = './data/validation/df_averages.csv'

def best_out_of_n_analysis(n:int):

    # load the model
    model, opt = load_model(conf_path)
    if torch.cuda.is_available():
        model = model.to('cuda')
    # read data
    lq_paths = fiFindByWildcard(os.path.join(dataroot_lr, '*.jpg'))
    gt_paths = fiFindByWildcard(os.path.join(dataroot_gt, '*.jpg'))
    lq_paths.sort()
    gt_paths.sort()

    # verify if lq_paths filenames and gt_paths filenames agree on all indices
    for i in range(len(lq_paths)):
        assert lq_paths[i].split('/')[-1] == gt_paths[i].split('/')[-1]

    lqs = [imread(p) for p in lq_paths]
    gts = [imread(p) for p in gt_paths]

    # instantiate measure
    measure = Measure.Measure()

    # initialize the temperatures to sample at
    temperatures = [x for x in np.linspace(0, 1, num=11)]
    filenames = [p.split('/')[-1] for p in lq_paths]

    # initialize empty dataframes to record the best out of n values at each temperature
    df_lpips = pd.DataFrame(index=filenames, columns=temperatures)
    df_psnr = pd.DataFrame(index=filenames, columns=temperatures)
    df_ssim = pd.DataFrame(index=filenames, columns=temperatures)

    # initialize arrays to record sums of measures at each temperature (later used to calc averages)
    sum_lpips, sum_psnr, sum_ssim = [0 for i in np.linspace(0, 1, num=11)], [0 for i in np.linspace(0, 1, num=11)],[0 for i in np.linspace(0, 1, num=11)]

    # count the current lq,gt pair
    count = 0

    for lq, gt, fname in tqdm(zip(lqs, gts, filenames)):
        print(f'Image#: {count + 1}')
        for i, temperature in enumerate(np.linspace(0, 1, num=11)):

            #initialize max and min values and images

            max_psnr, max_ssim,min_lpips = (-np.inf,-np.inf,np.inf)
            max_psnr_sr, max_ssim_sr, min_lpips_sr = None, None, None

            # generate n samples at each temperature
            for j in range(n):
                # Sample a super-resolution for a low-resolution image
                sr = rgb(model.get_sr(lq=t(lq), heat=temperature))
                psnr, ssim, lpips = measure.measure(sr, gt)
                print(
                    'Temperature: {:0.2f} - PSNR: {:0.1f}, SSIM: {:0.1f}, LPIPS: {:0.2f}\n\n'.format(temperature, psnr,
                                                                                                     ssim, lpips))
                if psnr > max_psnr:
                    max_psnr = psnr
                    max_psnr_sr = sr

                if ssim > max_ssim:
                    max_ssim = ssim
                    max_ssim_sr = sr

                if lpips < min_lpips:
                    min_lpips = lpips
                    min_lpips_sr = sr

            # record running sum of optimal values of measures
            sum_lpips[i] += min_lpips
            sum_psnr[i] += max_psnr
            sum_ssim[i] += max_ssim

            # record the values of the best measures for each temperature
            df_lpips[temperature][fname] = min_lpips
            df_psnr[temperature][fname] = max_psnr
            df_ssim[temperature][fname] = max_ssim

        count += 1

    # save the analysis data
    df_lpips.to_csv(df_lpips_save_path)
    df_ssim.to_csv(df_ssim_save_path)
    df_psnr.to_csv(df_psnr_save_path)

    # calculate average statisticts
    avg_psnr = [p / len(lq_paths) for p in sum_psnr]
    avg_ssim = [p / len(lq_paths) for p in sum_ssim]
    avg_lpips = [p / len(lq_paths) for p in sum_lpips]

    # save average statisticts
    with open(avg_psnr_save_path, 'wb') as f:
        pickle.dump(avg_psnr, f)
    with open(avg_lpips_save_path, 'wb') as f:
        pickle.dump(avg_lpips, f)
    with open(avg_ssim_save_path, 'wb') as f:
        pickle.dump(avg_ssim, f)

    # save average statisticts inside a single dataframe
    avg_psnr_series = pd.Series(data=avg_psnr)
    avg_ssim_series = pd.Series(data=avg_ssim)
    avg_lpips_series = pd.Series(data=avg_lpips)

    dataframe_dict = {'psnr': avg_psnr_series, 'ssim': avg_ssim_series, 'lpips': avg_lpips_series}
    avg_stats_df = pd.DataFrame(dataframe_dict)
    avg_stats_df.to_csv(df_averages_save_path)


def generateOldNewGraphsAll():
    generateGraphOldvNew(df_lpips_save_path,df_ssim_save_path,df_psnr_save_path,df_lpips_save_path_old,df_ssim_save_path_old, df_psnr_save_path_old)

if __name__ == '__main__':
    #best_out_of_n_analysis(3)
    generateOldNewGraphsAll()