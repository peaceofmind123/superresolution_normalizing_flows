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

temp_vs_psnr_save_path = './data/validation/temp-psnr.png'
temp_vs_ssim_save_path = './data/validation/temp-ssim.png'
temp_vs_lpips_save_path = './data/validation/temp-lpips.png'

lpips_vs_psnr_save_path = './data/validation/lpips-psnr.png'
lpips_vs_ssim_save_path = './data/validation/lpips-ssim.png'
psnr_vs_ssim_save_path = './data/validation/psnr-ssim.png'

lpips_nsamples_save_path = './data/validation/lpips-nsamples.png'

def result_analysis():

    # load the model
    model, opt = load_model(conf_path)

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

    # instantiate empty dataframes with columns as temperatures and rows as filename
    temperatures = [x for x in np.linspace(0, 1, num=11)]
    filenames = [p.split('/')[-1] for p in lq_paths]
    df_lpips = pd.DataFrame(index=filenames, columns=temperatures)
    df_psnr = pd.DataFrame(index=filenames, columns=temperatures)
    df_ssim = pd.DataFrame(index=filenames, columns=temperatures)

    # initialize arrays to record sums of measures at each temperature (later used to calc averages)
    sum_lpips,sum_psnr, sum_ssim = [0 for i in np.linspace(0, 1, num=11)],[0 for i in np.linspace(0, 1, num=11)],[0 for i in np.linspace(0, 1, num=11)]

    # count the current lq,gt pair
    count = 0

    for lq,gt,fname in tqdm(zip(lqs,gts,filenames)):
        print(f'Image#: {count+1}')
        for i,temperature in enumerate(np.linspace(0, 1, num=11)):
            # Sample a super-resolution for a low-resolution image
            sr = rgb(model.get_sr(lq=t(lq), heat=temperature))
            if count == 0 and i ==0:
                plt.imsave('sr.png',sr)
                plt.imsave('gt.png',gt)
            psnr, ssim, lpips = measure.measure(sr, gt)
            print('Temperature: {:0.2f} - PSNR: {:0.1f}, SSIM: {:0.1f}, LPIPS: {:0.2f}\n\n'.format(temperature, psnr,
                                                                                                                ssim, lpips))
            sum_lpips[i] += lpips
            sum_psnr[i] += psnr
            sum_ssim[i] += ssim

            df_lpips[temperature][fname] = lpips
            df_psnr[temperature][fname] = psnr
            df_ssim[temperature][fname] = ssim
        count+=1
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
        pickle.dump(avg_psnr,f)
    with open(avg_lpips_save_path,'wb') as f:
        pickle.dump(avg_lpips, f)
    with open(avg_ssim_save_path,'wb') as f:
        pickle.dump(avg_ssim,f)

    # save average statisticts inside a single dataframe
    avg_psnr_series = pd.Series(data=avg_psnr)
    avg_ssim_series = pd.Series(data=avg_ssim)
    avg_lpips_series = pd.Series(data=avg_lpips)

    dataframe_dict = {'psnr':avg_psnr_series, 'ssim':avg_ssim_series, 'lpips':avg_lpips_series}
    avg_stats_df = pd.DataFrame(dataframe_dict)
    avg_stats_df.to_csv(df_averages_save_path)

# generate average statistics and graphs from the analysis data
def generateGraphs(df_lpips_path, df_ssim_path, df_psnr_path):
    df_lpips = pd.read_csv(df_lpips_path, index_col=0)
    df_ssim = pd.read_csv(df_ssim_path,index_col=0)
    df_psnr = pd.read_csv(df_psnr_path,index_col=0)

    mean_lpips_vals = df_lpips.mean(axis=0)
    mean_ssim_vals = df_ssim.mean(axis=0)
    mean_psnr_vals = df_psnr.mean(axis=0)

    # generate Temperature vs metrics graphs
    temperatures = np.linspace(0, 1, num=11)
    generateGraph('Temperature', 'PSNR',temp_vs_psnr_save_path,temperatures,mean_psnr_vals,1)
    generateGraph('Temperature', 'SSIM', temp_vs_ssim_save_path, temperatures, mean_ssim_vals,2)
    generateGraph('Temperature', 'LPIPS', temp_vs_lpips_save_path, temperatures, mean_lpips_vals,3)

    # generate metric vs metric graphs
    generateGraph('LPIPS', 'PSNR', lpips_vs_psnr_save_path, mean_lpips_vals, mean_psnr_vals,4,True)
    generateGraph('LPIPS', 'SSIM', lpips_vs_ssim_save_path, mean_lpips_vals, mean_ssim_vals,5,True)
    generateGraph('PSNR', 'SSIM', psnr_vs_ssim_save_path, mean_psnr_vals, mean_ssim_vals,6)


def generateGraph(xlabel, ylabel, save_path, xs,ys, fig_idx, inv=False):

    plt.figure(fig_idx)
    _, ax = plt.subplots(1,1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tick_params(
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off,
        left=False,
        right=False)
    ax.plot(xs,ys,marker='o')
    if inv:
        ax.invert_xaxis() # Forgot to include this line of code, causing a bug

    plt.savefig(fname=save_path, dpi=600, bbox_inches='tight')

# utility functions
def find_files(wildcard): return natsort.natsorted(glob.glob(wildcard, recursive=True))


def pickleRead(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Convert to tensor
def t(array):
    x = torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255
    if torch.cuda.is_available():
        return x.to(device='cuda')
    return x
# convert to image
def rgb(t): return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)



if __name__ == '__main__':
    generateGraphs(df_lpips_path=df_lpips_save_path,df_ssim_path=df_ssim_save_path, df_psnr_path=df_psnr_save_path)
    #result_analysis()
