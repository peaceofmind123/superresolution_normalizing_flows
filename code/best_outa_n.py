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
from result_analysis import find_files, pickleRead, t, rgb, generateGraphOldvNew, generateNCurves

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 22
conf_path = './confs/SRFlow_CelebA_8X.yml'

dataroot_lr = './data/validation/lr'
dataroot_gt = './data/validation/hr'

df_lpips_save_path = './data/validation/df_lpips.csv'
df_ssim_save_path = './data/validation/df_ssim.csv'
df_psnr_save_path = './data/validation/df_psnr.csv'

df_lpips_save_path_old = './data/validation/df_lpips_old.csv'
df_ssim_save_path_old = './data/validation/df_ssim_old.csv'
df_psnr_save_path_old = './data/validation/df_psnr_old.csv'

avg_psnr_save_path = './data/validation/avg_psnr.pkl'
avg_lpips_save_path = './data/validation/avg_lpips.pkl'
avg_ssim_save_path = './data/validation/avg_ssim.pkl'

df_averages_save_path = './data/validation/df_averages.csv'

df_averages_save_path_psnr = './data/validation/df_averages_psnr.csv'
df_averages_save_path_ssim = './data/validation/df_averages_ssim.csv'
df_averages_save_path_lpips = './data/validation/df_averages_lpips.csv'


class Analysis:
    def __init__(self, conf_path, dataroot_lr, dataroot_gt):

        # load the model
        opt = option.parse(conf_path, is_train=False)
        print(f'opt: {opt}')
        opt['gpu_ids'] = None
        opt = option.dict_to_nonedict(opt)
        print(f'opt_after_dict_nonedict: {opt}')
        self.model = create_model(opt)
        print(f'model initial: {self.model}')
        model_path = opt_get(opt, ['model_path'], None)
        print(f'model_path: {model_path}')
        self.model.load_network(load_path=model_path, network=self.model.netG)
        print(f'model before returning: {self.model}')

        self.opt = opt



        # read data
        self.lq_paths = fiFindByWildcard(os.path.join(dataroot_lr, '*.jpg'))
        self.gt_paths = fiFindByWildcard(os.path.join(dataroot_gt, '*.jpg'))
        self.lq_paths.sort()
        self.gt_paths.sort()

        # verify if lq_paths filenames and gt_paths filenames agree on all indices
        for i in range(len(self.lq_paths)):
            assert self.lq_paths[i].split('/')[-1] == self.gt_paths[i].split('/')[-1]

        self.lqs = [imread(p) for p in self.lq_paths]
        self.gts = [imread(p) for p in self.gt_paths]

        # instantiate measure
        self.measure = Measure.Measure()

        # initialize the temperatures to sample at
        self.temperatures = [x for x in np.linspace(0, 1, num=11)]
        self.filenames = [p.split('/')[-1] for p in self.lq_paths]

        # dataframes and sums
        self.dataframes_lpips = None
        self.dataframes_psnr = None
        self.dataframes_ssim = None

        self.avgs_lpips = None
        self.avgs_psnr = None
        self.avgs_ssim = None

    # initialize an array of dataframes for multiple n-sample analysis
    def initializeDataFrames(self, n_samples_start, n_samples_end):
        self.dataframes_lpips = {i: pd.DataFrame(index=self.filenames, columns=self.temperatures) for i in
                                 range(n_samples_start, n_samples_end + 1)}
        self.dataframes_psnr = {i: pd.DataFrame(index=self.filenames, columns=self.temperatures) for i in
                                range(n_samples_start, n_samples_end + 1)}
        self.dataframes_ssim = {i: pd.DataFrame(index=self.filenames, columns=self.temperatures) for i in
                                range(n_samples_start, n_samples_end + 1)}

    # initialize sums for multiple n-sample analysis
    def initializeRunningAvgs(self, n_samples_start, n_samples_end):
        self.avgs_lpips = {i: [0 for _ in np.linspace(0, 1, num=11)] for i in range(n_samples_start, n_samples_end + 1)}
        self.avgs_ssim = {i: [0 for _ in np.linspace(0, 1, num=11)] for i in range(n_samples_start, n_samples_end + 1)}
        self.avgs_psnr = {i: [0 for _ in np.linspace(0, 1, num=11)] for i in range(n_samples_start, n_samples_end + 1)}

    # calculate running average
    def calcRunningAvg(self, prev_val, current_val, current_count):
        # TEST PASSING
        assert current_count > 0
        return (prev_val * (current_count - 1) + current_val) / current_count

    # save avg stats inside a single dataframe
    def saveAvgStats(self, df_averages_save_path_psnr,df_averages_save_path_ssim,df_averages_save_path_lpips, avgs_psnr, avgs_ssim, avgs_lpips, n_samples_start, n_samples_end):
        # TEST PASSING
        # note that the avgs_psnr,etc are provided as args for testability without
        # running the analysis first

        avg_psnr_df = {sample_size: pd.Series(data=avgs_psnr[sample_size], index=np.linspace(0,1,num=11)) for sample_size in
                           range(n_samples_start, n_samples_end + 1)}
        avg_lpips_df = {sample_size: pd.Series(data=avgs_lpips[sample_size], index=np.linspace(0,1,num=11)) for sample_size in
                            range(n_samples_start, n_samples_end + 1)}
        avg_ssim_df = {sample_size: pd.Series(data=avgs_ssim[sample_size], index=np.linspace(0,1,num=11)) for sample_size in
                           range(n_samples_start, n_samples_end + 1)}

        pd.DataFrame(avg_psnr_df).to_csv(df_averages_save_path_psnr)
        pd.DataFrame(avg_ssim_df).to_csv(df_averages_save_path_ssim)
        pd.DataFrame(avg_lpips_df).to_csv(df_averages_save_path_lpips)

    def generateGraphsFromCSVs(self, df_averages_save_path_psnr,
                               df_averages_save_path_ssim,
                               df_averages_save_path_lpips,
                               save_path_dict):
        dfs = self.readAveragesDFs([df_averages_save_path_psnr, df_averages_save_path_ssim,df_averages_save_path_lpips])
        # create legend
        legends = [f'n={j + 1}' for j in range(len(dfs['psnr']))]

        # initialize figure id
        fig_id = 0
        # get ys and xs from the dataframes
        for meas in ['psnr','ssim','lpips']: # 3 for psnr, ssim and lpips
            # construct to hold the ys and xs values for multiple curves
            yss = []
            xs_temp = np.linspace(0, 1, 11)
            xss_temp = [xs_temp for i in range(10)]

            # fill the constructs
            df = dfs[meas] # get the df for a particular measure
            for sample_size in df: # the column indices are sample_sizes
                series = df[sample_size]
                ys = series.to_numpy()
                yss.append(ys)

            # draw the curves
            # temperature vs metrics curves
            generateNCurves('Temperature', meas,
                            save_path_dict['temperature'][meas],
                            xss_temp, yss, fig_id, legends=legends)
            fig_id += 1
            # metric vs metric curves
            for meas2 in ['psnr','ssim','lpips']:
                if (meas =='lpips' and meas2 == 'psnr') or \
                        (meas == 'lpips' and meas2 == 'ssim') \
                        or (meas == 'psnr' and meas2 == 'ssim'):
                    not_to_inv = (meas =='psnr' and meas2 == 'ssim')
                    df = dfs[meas2]
                    yss2 = [df[sample_size].to_numpy() for sample_size in df]
                    generateNCurves(meas, meas2,
                                    save_path_dict[meas][meas2],
                                    yss, yss2, fig_id,
                                    legends=legends, inv=(not not_to_inv))
                    fig_id += 1

    def readAveragesDFs(self,dfs_paths):

        return {meas: pd.read_csv(p, index_col=0) for meas,p in zip(['psnr', 'ssim','lpips'],
                dfs_paths)}

    def best_out_of_n_analysis(self, n_samples_start, n_samples_end ):
        # initialize dataframes and sums
        self.initializeDataFrames(n_samples_start, n_samples_end)
        self.initializeRunningAvgs(n_samples_start, n_samples_end)

        # count number of current image
        count_img = 0
        for lq, gt, fname in tqdm(zip(self.lqs, self.gts, self.filenames), total=len(self.lqs)):
            for i, temperature in enumerate(np.linspace(0, 1, num=11)):

                # for recording the values of psnr, ssim and lpips
                psnrs, ssims, lpips_vals = [], [], []

                # for recording sum and average values

                # generate the samples and record the measures
                for j in range(n_samples_start, n_samples_end + 1):
                    # Sample a super-resolution for a low-resolution image
                    sr = rgb(self.model.get_sr(lq=t(lq), heat=temperature))
                    psnr, ssim, lpips = self.measure.measure(sr, gt)
                    psnrs.append(psnr)
                    ssims.append(ssim)
                    lpips_vals.append(lpips)

                # compute the maxes and mins at each sample size
                for sample_size in range(n_samples_start, n_samples_end + 1):
                    max_psnr = max(psnrs[0:sample_size])
                    max_ssim = max(ssims[0:sample_size])
                    min_lpips = min(lpips_vals[0:sample_size])

                    # compute running avgs at each sample size and each temperature(i) across multiple images

                    self.avgs_ssim[sample_size][i] = self.calcRunningAvg(self.avgs_ssim[sample_size][i], max_ssim,
                                                                         count_img + 1)
                    self.avgs_psnr[sample_size][i] = self.calcRunningAvg(self.avgs_psnr[sample_size][i], max_psnr,
                                                                         count_img + 1)
                    self.avgs_lpips[sample_size][i] = self.calcRunningAvg(self.avgs_lpips[sample_size][i], min_lpips,
                                                                          count_img + 1)

            # update image count
            count_img += 1


        # save the average stats
        self.saveAvgStats(df_averages_save_path_psnr,df_averages_save_path_ssim,df_averages_save_path_lpips,
                          self.avgs_psnr, self.avgs_ssim, self.avgs_lpips,n_samples_start, n_samples_end)


def best_out_of_n_analysis(n: int, df_averages_save_path=df_averages_save_path):
    # load the model
    model, opt = load_model(conf_path)
    print(f'Model is none: {model is None}')
    if torch.cuda.is_available():
        model = model.to(device=torch.device('cuda'))

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
    sum_lpips, sum_psnr, sum_ssim = [0 for i in np.linspace(0, 1, num=11)], [0 for i in np.linspace(0, 1, num=11)], [0
                                                                                                                     for
                                                                                                                     i
                                                                                                                     in
                                                                                                                     np.linspace(
                                                                                                                         0,
                                                                                                                         1,
                                                                                                                         num=11)]

    # count the current lq,gt pair
    count = 0

    for lq, gt, fname in tqdm(zip(lqs, gts, filenames), total=len(lqs)):
        # print(f'Image#: {count + 1}')
        for i, temperature in enumerate(np.linspace(0, 1, num=11)):

            # initialize max and min values and images

            max_psnr, max_ssim, min_lpips = (-np.inf, -np.inf, np.inf)
            max_psnr_sr, max_ssim_sr, min_lpips_sr = None, None, None

            # generate n samples at each temperature
            for j in range(n):
                # Sample a super-resolution for a low-resolution image
                sr = rgb(model.get_sr(lq=t(lq), heat=temperature))
                psnr, ssim, lpips = measure.measure(sr, gt)
                # print(
                #    'Temperature: {:0.2f} - PSNR: {:0.1f}, SSIM: {:0.1f}, LPIPS: {:0.2f}\n\n'.format(temperature, psnr,
                #                                                                                     ssim, lpips))
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
    generateGraphOldvNew(df_lpips_save_path, df_ssim_save_path, df_psnr_save_path, df_lpips_save_path_old,
                         df_ssim_save_path_old, df_psnr_save_path_old)


def runAnalysis(start=1, end=5):
    """
    Run best out of n analysis for multiple values of n
    """
    for i in range(start, end + 1):
        best_out_of_n_analysis(i, f'./data/validation/df_averages_{i}_samples.csv')


def runAnalysisAllAtOnce(end=5):
    pass


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    print(f'opt: {opt}')
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    print(f'opt_after_dict_nonedict: {opt}')
    model = create_model(opt)
    print(f'model initial: {model}')
    model_path = opt_get(opt, ['model_path'], None)
    print(f'model_path: {model_path}')
    model.load_network(load_path=model_path, network=model.netG)
    print(f'model before returning: {model}')
    if torch.cuda.is_available():
        return model.to(device=torch.device('cuda')), opt
    return model, opt


def runSaveDfsTest():
    # successful
    analysis = Analysis(conf_path, dataroot_lr, dataroot_gt)
    n_samples_start = 1
    n_samples_end = 5
    avgs_lpips = {i: np.random.rand(11) for i in range(n_samples_start, n_samples_end + 1)}
    avgs_ssim = {i: np.random.rand(11) for i in range(n_samples_start, n_samples_end + 1)}
    avgs_psnr = {i: np.random.rand(11) for i in range(n_samples_start, n_samples_end + 1)}

    analysis.saveAvgStats(df_averages_save_path_psnr, df_averages_save_path_ssim, df_averages_save_path_lpips,avgs_psnr,avgs_ssim,avgs_lpips,n_samples_start,n_samples_end)


def runCalcRunningAvgTest():
    # successful test
    analysis = Analysis(conf_path, dataroot_lr, dataroot_gt)
    for i in range(10):
        ls = np.random.rand(10)
        avg = ls.mean(axis=0)

        running_avg = 0
        for j in range(10):
            running_avg = analysis.calcRunningAvg(running_avg, ls[j], j+1)

        assert f'{avg:.2f}' == f'{running_avg:.2f}'
    print("Test Passing!!")

def classInitializationTest():
    """Meant to be run in debug mode to observe whether the variables are init correctly"""
    analysis = Analysis(conf_path,dataroot_lr, dataroot_gt)
    pass

def endToEndTest():
    test_dataroot_lr = './data/validation/test/lr'
    test_dataroot_gt = './data/validation/test/gt'

    analysis = Analysis(conf_path,test_dataroot_lr, test_dataroot_gt)
    analysis.best_out_of_n_analysis(1,5)

def generateCurvesTest():
    test_dataroot_lr = './data/validation/test/lr'
    test_dataroot_gt = './data/validation/test/gt'
    root_path = './data/validation/'
    save_path_dict = {'temperature': {
        'psnr': root_path+'temp-psnr.png',
        'ssim': root_path+'temp-ssim.png',
        'lpips':root_path + 'temp-lpips.png'
    },'lpips': {
        'psnr': root_path+'lpips-psnr.png',
        'ssim': root_path + 'lpips-ssim.png'
    },'psnr': {
        'ssim': root_path + 'psnr-ssim.png'
    }
    }
    analysis = Analysis(conf_path, test_dataroot_lr, test_dataroot_gt)
    analysis.generateGraphsFromCSVs(df_averages_save_path_psnr,
                                    df_averages_save_path_ssim,
                                    df_averages_save_path_lpips,
                                    save_path_dict)
def runTests():
    runSaveDfsTest()
    runCalcRunningAvgTest()


def generateCurvesMain():
    analysis = Analysis(conf_path,dataroot_lr, dataroot_gt)
    root_path = './data/validation/'
    save_path_dict = {'temperature': {
        'psnr': root_path + 'temp-psnr.png',
        'ssim': root_path + 'temp-ssim.png',
        'lpips': root_path + 'temp-lpips.png'
    }, 'lpips': {
        'psnr': root_path + 'lpips-psnr.png',
        'ssim': root_path + 'lpips-ssim.png'
    }, 'psnr': {
        'ssim': root_path + 'psnr-ssim.png'
    }
    }
    analysis.generateGraphsFromCSVs(df_averages_save_path_psnr,
                                    df_averages_save_path_ssim,
                                    df_averages_save_path_lpips,
                                    save_path_dict)

if __name__ == '__main__':
    # best_out_of_n_analysis(3)
    # generateOldNewGraphsAll()
    #runAnalysis()
    #runTests()
    #runCalcRunningAvgTest()
    #endToEndTest()
    generateCurvesMain()