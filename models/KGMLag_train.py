# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:49:28 2022

@author: yang8460

torch.Dataloader, batch 64, 21s/per 10 step
torch.Dataloader, batch 512, 72s/per 10 step
torch.Dataloader, batch 512,8 workers, 80s/per 30 step
"""

import torch
import torch.nn as nn
import torch.optim as optim
import ECOSOC_util as util
import ECOSOC_networks as net
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
# import ECONET_dataPrepare_WarmingUp as E_data
import os
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")
 
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



def plot_test_scatter(p,o,n=None,outFolder=None,saveFig=False,note='',title=''):
    fig, ax = plt.subplots(1, 1,figsize = (6,5))
    if n==None:
        x=np.array(o)
        y=np.array(p)
    else:
        x=np.array(o[n[0]:n[1]])
        y=np.array(p[n[0]:n[1]])
    plt.scatter(x, y, 
                  color='b',  label='')
    R2 = np.corrcoef(x, y)[0, 1] ** 2
    RMSE = (np.sum((x - y) ** 2) / len(y)) ** 0.5
    plt.text(0.05, 0.87, r'$R^2 $ = %.3f'%R2, transform=ax.transAxes,fontsize=16)
    plt.text(0.05, 0.80, r'$RMSE $ = %.3f'%RMSE, transform=ax.transAxes,fontsize=16)
    plt.text(0.05, 0.73, r'$n $ = %d'%len(x), transform=ax.transAxes,fontsize=16)
    plt.xlabel('Observed', fontsize=14)
    plt.ylabel('Predicted', fontsize=14)
    lim = np.max([np.max(x),np.max(y)])
    plt.plot(np.arange(0,np.ceil(lim)+1), np.arange(0,np.ceil(lim)+1), 'k', label='1:1 line')
    plt.xlim([0,lim])
    plt.ylim([0,lim])
    
    if saveFig:
        if n==None:
            plt.title('%s samples all'%title)
            fig.savefig('%s/test_scatter_%s.png'%(outFolder,note))
        else:
            plt.title('%s samples from %d to %d'%(title,n[0],n[1]))
            fig.savefig('%s/test_scatter_%d-%d_%s.png'%(outFolder,n[0],n[1],note), dpi=500)
    else:
        plt.title(title)

def slope_np(x,y):
    xyMean = np.mean(x*y)
    xMean = np.mean(x)
    yMean = np.mean(y)
    x2Mean = np.mean(x**2)
    return (xyMean-xMean*yMean)/(x2Mean-xMean**2 + 10e-6)

def plotScatterDense(x_, y_, alpha=1, binN=200, thresh_p=None, outFolder='',
                     saveFig=False, note='', title='', uplim=None, downlim=None, auxText=None, legendLoc=4):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.rcParams.update({'font.size': 15})
    x_ = np.array(x_)
    y_ = np.array(y_)
    if len(y_) > 1:

        # Calculate the point density
        if not (thresh_p is None):
            thresh = (np.max(np.abs(x_)) * thresh_p)

            x = x_[((x_ > thresh) | (x_ < -thresh)) & ((y_ > thresh) | (y_ < -thresh))]
            y = y_[((x_ > thresh) | (x_ < -thresh)) & ((y_ > thresh) | (y_ < -thresh))]
        else:
            x = x_
            y = y_

        tmp = stats.linregress(x, y)
        para = [tmp[0], tmp[1]]
        # para = np.polyfit(x, y, 1)   # can't converge for large dataset
        y_fit = np.polyval(para, x)  #
        plt.plot(x, y_fit, 'b')

    # histogram definition
    bins = [binN, binN]  # number of bins

    # histogram the data
    hh, locx, locy = np.histogram2d(x, y, bins=bins)

    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(x, y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]

    plt.scatter(x2, y2, c=z2, cmap='Blues', marker='.', alpha=alpha, vmin=0, vmax=800)

    if uplim == None:
        uplim = 1.2 * max(np.hstack((x, y)))
    if downlim == None:
        downlim = 0 # 0.8 * min(np.hstack((x, y)))


    plt.xlabel('Synthetic data', fontsize=16)
    plt.ylabel('Predictions', fontsize=16)
    # plt.legend(loc=1)
    plt.axis('square')
    if note == 'NEE':
        downlim = -20
    elif note == 'ET':
        uplim = 10
    elif note == 'SWC':
        uplim = 0.8
    elif note == 'Yld':
        uplim = 800
    elif note == 'GPP':
        uplim = 30
    figRange = uplim - downlim
    plt.xlim([downlim, uplim])
    plt.ylim([downlim, uplim])
    plt.plot(np.arange(downlim, np.ceil(uplim) + 1), np.arange(downlim, np.ceil(uplim) + 1), 'k', label='1:1 line')
    if not legendLoc is None:
        if legendLoc == False:
            plt.legend(edgecolor='w', facecolor='w', fontsize=12)
        else:
            plt.legend(loc=legendLoc, edgecolor='w', facecolor='w', fontsize=12, framealpha=0)
    plt.title(title, y=0.9)

    if len(y) > 1:
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
        # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
        plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2 $= ' + str(R2)[:5], fontsize=14)
        # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
        plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE $= ' + str(RMSE)[:5], fontsize=14)

        plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Slope $= ' + str(para[0])[:5], fontsize=14)
    if not auxText == None:
        plt.text(0.05, 0.9, auxText, transform=ax.transAxes, fontproperties='Times New Roman', fontsize=20)
    plt.colorbar()

    if saveFig:
        plt.title('%s' % title)
        fig.savefig('%s/test_scatter_%s.png' % (outFolder, note))

    else:
        plt.title(title)
            
def plot_test_series(p,o,n=None, year=None, outFolder=None,saveFig=False,note='',title=''):

    x=np.array(o)
    y=np.array(p)

    fig = plt.figure(figsize=(10,5))
    plt.plot(y,
                  color='r',  label='predicted')
    plt.plot(x,
                  color='y',  label='synthetic')
    plt.legend()
    if saveFig:
        if n==None:
            plt.title('%s'%title)
            fig.savefig('%s/test_series_%s.png'%(outFolder,note))
        else:
            plt.title('%s: site %d year %d'%(title, n, year+2000))
            fig.savefig('%s/test_series_site%d_year%d_%s.png'%(outFolder, n, year+2000,note), dpi=300)
    else:
        plt.title(title)

def plot_test_singleStep(pre,obs,n):
    plt.figure()
    plt.scatter(np.squeeze(obs)[:n], np.squeeze(pre)[:n], 
                  color='b',  label='')
    plt.xlabel('Synthetic data', fontsize=14)
    plt.ylabel('Prediction', fontsize=14)
    plt.figure(figsize=(10,5))
    plt.plot(np.squeeze(pre)[:n], 
                  color='r',  label='predicted')
    plt.plot(np.squeeze(obs)[:n],
                  color='y',  label='Synthetic')
    plt.legend()
    
    
if __name__ == '__main__':
    
    # hyper parameters
    batch_size = 256 # 256
    hidden_dim = 64 # 64
    n_epochs = 80 #100
    learning_rate = 1e-3
    # weight_decay = 1e-6
    lr_decay = 0.9
    saveResult = True
    
    # y_selectFeatures = ['DVS','GPP_daily','Biomass','LAI','GrainYield',]
    # y_NormCoef = [0.5,0.02,0.001,0.1,0.0015]
    X_selectFeatures = ['Tmax', 'Tmin', 'Humid', 'P', 'NSR', 'WS', # 30 variables
                        'Growseason','GrowseasonCC', 'Crop', 'Mixrate', 'N_rate', 'Harvest', # 'Covercrop', 
                        # 'VCMX','CHL4','GROUPX','STMX','GFILL','SLA1','VCMX_CC','CHL4_CC','GROUPX_CC','STMX_CC','GFILL_CC','SLA1_CC',
                        'PH', 'BD', 'Sand', 'Silt', 'FC', 'WP', 'Ks', 'gSOC']
    y_selectFeatures = ['DVS',
                        'ET','GPP','SWC',
                        'Biomass','Reco','NEE',
                        'LAI','Yld']

                  
    y_NormCoef = [0.5,
                  0.15, 0.02, 1.5,
                  0.001, 0.06, -0.05,
                  0.1,0.0015]
                  
    
    output_dim=[1,3,3,2]
    
    mode = 'paraPheno_c2'#'cropPheno_c2'#'para' #'crop'
    # mode = 'cropPheno_c2'
    modelPath = 'model'
    modelName = 'gru' # 'lstm', 'rnn', or 'gru'
    note = 'cornBelt_4cell_v2_%s'
    # note = '5000points_4cell_v1_2_RecoCorrected_%s'%mode
    now = datetime.now().strftime('%y%m%d')
    projectName = '%s-epoch%d-batch%d-%s'%(modelName,n_epochs,batch_size,now)
    outFolder = 'log/%s'%(projectName)

    # load dataset
    # split dataset
    # dataPath = r'F:\MidWest_counties\output_cornBelt20299_pkl_all_separate'
    dataPath = '/panfs/jay/groups/15/jinzn/yang6956/SOC_KGML/outputs_cornbelt_2403_clean/'
    # dataPath = r'F:/randomPoints_3I_5133_noH_v2_line3_pkl_v2_all_separate'
    siteYearList = [t.split('/')[-1].split('.')[0] for t in glob.glob('%s/*.pickle'%dataPath)]
    siteYearList = ['%s/%s.pickle' % (dataPath, t) for t in siteYearList]
    # inputList = ['%s/input_%s.pkl'%(dataPath,t) for t in siteYearList]
    #     # outputList = ['%s/output_%s.pkl'%(dataPath,t) for t in siteYearList]
    
    X_val, X_train, X_test = util.train_val_test_split_site(dataList=siteYearList, test_ratio=0.1)
    print(len(siteYearList), len(X_train), len(X_val), len(X_test), flush=True)
    # get the dataloader ready

    length = len(pd.date_range('2000-01-01', '2020-12-31', freq='D').tolist())
    train_ds = util.EcoNet_dataset_pkl_yearly(X_train,X_selectFeatures=X_selectFeatures, y_selectFeatures=y_selectFeatures,y_NormCoef=y_NormCoef, length=length)
    eval_ds = util.EcoNet_dataset_pkl_yearly(X_val,X_selectFeatures=X_selectFeatures, y_selectFeatures=y_selectFeatures,y_NormCoef=y_NormCoef, length=length)
    test_ds = util.EcoNet_dataset_pkl_yearly(X_test,X_selectFeatures=X_selectFeatures, y_selectFeatures=y_selectFeatures,y_NormCoef=y_NormCoef, length=length)

    train_dl = DataLoader(train_ds,batch_size=batch_size, shuffle=True) # num_workers=8
    eval_dl = DataLoader(eval_ds,batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    # train_dl = DeviceDataLoader(train_dl, device)
    # test_dl = DeviceDataLoader(test_dl, device)
    # test_dl_one = DeviceDataLoader(test_dl_one, device)
    
    # b0,b1,l = train_ds[1]
    # for a in train_dl:
    #     break
    # a00,a10,l0 = train_dl0.getbatch()
    
    
    ## define the network
    input_dim = len(X_selectFeatures)
    # model = net.GRUModel_hiera_4cell_add_noise_cropType_pheno(input_dim=input_dim,hidden_dim=hidden_dim,
    #                               output_dim=output_dim)
    model = net.GRUModel_hiera_4cell_add_noise_parameters_3layerDecoder_v2(input_dim=input_dim,hidden_dim=hidden_dim,
                                  output_dim=output_dim,mode=mode)
    model = to_device(model, device) # model.load_state_dict(torch.load('model/lstm-epoch10-batch64-99points_state_dict.pth'))
    
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Decay LR by a factor every epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=lr_decay)
    
    # train
    opt = util.Optimization_decoder(model=model, loss_fn=loss_fn, optimizer=optimizer,exp_lr_scheduler=exp_lr_scheduler, hiera=True)
    opt.train_yearly(train_dl, eval_dl, n_epochs=n_epochs, n_features=input_dim, interv = 30, timeCount=True)
    
    if saveResult:     
        if not os.path.exists('log'):
            os.mkdir('log') 
        if not os.path.exists(outFolder):
            os.mkdir(outFolder)
        if not os.path.exists(modelPath):
            os.mkdir(modelPath)
            
    opt.plot_losses(outFolder,saveFig=saveResult)

    # save model
    if saveResult:     
        
        torch.save(model.state_dict(), '%s/%s_state_dict.pth'%(modelPath,projectName))
        torch.save(model, '%s/%s.pth'%(modelPath,projectName))
        losses = pd.DataFrame()
        losses['train_losses'] = opt.train_losses
        
        losses['val_losses'] = opt.val_losses
        losses.to_csv('%s/train_losses_log.csv'%(outFolder))
        
    # test  
    predictions, values = opt.evaluate_yearly(test_dl, batch_size=batch_size, n_features=input_dim)



    for n in range(len(predictions)):
        if n == 0:
            predict = predictions[0]
            synthetic = values[0]
        else:
            predict = np.concatenate((predict, predictions[n]), axis = 0)
            synthetic = np.concatenate((synthetic, values[n]), axis = 0)

units = ['', '(mm)', r'(gC/m$^2$/day)', '(0-30 cm)', r'(gC/m$^2$)', r'(gC/m$^2$/day)', r'(gC/m$^2$/day)', r'(m$^2$/m$^2$)', r'(gC/m$^2$)']
titles = ['DVS', 'ET', 'GPP', 'SWC', 'Biomass', 'Reco', 'NEE', 'LAI', 'Grain Yield']
    # pick 10 sites to plot time series
    np.random.seed(10)
    sites = np.random.randint(0, predict.shape[0]//21, 10)
    
    for i, sf in enumerate(y_selectFeatures):
        p = predict[:,:,i].flatten()  / y_NormCoef[i]
        o = synthetic[:,:,i].flatten()  / y_NormCoef[i]
        print(sf, len(p))

        if saveResult:
            data = pd.DataFrame()
            data['predictions'] = p
            data['observations'] = o
            data.to_csv('%s/test_series_%s.csv' % (outFolder, sf))

        plotScatterDense(x_=o, y_=p, outFolder=outFolder, saveFig=saveResult, note=sf, title=sf + ' ' +units[i])

        for site_i in sites:
            years = np.random.randint(0, 20, 2)
            p_site = (predict[site_i*21:(site_i+1)*21,:,i] / y_NormCoef[i]).flatten()
            o_site = (synthetic[site_i*21:(site_i+1)*21,:,i] / y_NormCoef[i]).flatten()
            for year in years:
                p_year = p_site[(year*365+120):(year*365+365+120)]
                o_year = o_site[(year*365+120):(year*365+365+120)]
                plot_test_series(p=p_year, o=o_year, n=site_i, year=year, outFolder=outFolder, saveFig=saveResult, note=sf, title=sf)
