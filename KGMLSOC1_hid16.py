# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 2023
@author: yangyufeng
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import ECOSOC_util as util
# import ECONET_Networks as netDA
import ECOSOC_networks as netSOC
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import r2_score
# import ECONET_dataPrepare_WarmingUp as E_data
import os
import glob
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def Z_norm(X):
    X_mean=X.mean()
    X_std=np.std(np.array(X))
    return (X-X_mean)/X_std, X_mean, X_std

def Z_norm_reverse(X,Xscaler):
    return (X*Xscaler[1]+Xscaler[0])

class R2Loss(nn.Module):
    #calculate coefficient of determination
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - torch.nn.functional.mse_loss(y_pred, y, reduction="mean") / var_y

def changePadValue(y, seq_length):
    # replace the padding to -999
    t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                batch_first=True, enforce_sorted=False)
    # y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True, padding_value=-999)
    return t


# inputs and outputs of KGML SOC model
X_Features = ['Tmax', 'Tmin', 'Humid', 'P', 'NSR', 'WS', # 20 variables
                    'Growseason', 'GrowseasonCC', 'Crop', 'Mixrate', 'N_rate', 'Harvest',
                    'PH', 'BD', 'Sand', 'Silt', 'FC', 'WP', 'Ks', 'gSOC']
y_Features = ['Yld', 'TSOC']

# inputs and outputs for the DA model
X_selectFeatures = ['Tmax', 'Tmin', 'Humid', 'P', 'NSR', 'WS', # 20 variables
                    'Growseason', 'GrowseasonCC', 'Crop', 'Mixrate', 'N_rate', 'Harvest',
                    'PH', 'BD', 'Sand', 'Silt', 'FC', 'WP', 'Ks', 'gSOC']
y_selectFeatures = ['DVS',
                    'ET', 'GPP', 'SWC',
                    'Biomass', 'Reco', 'NEE',
                    'LAI', 'Yld']


# load synthetic data
data_path ='/panfs/jay/groups/15/jinzn/yang6956/SOC_KGML/outputs_cornbelt_2403/'
# dataPath = r'F:/randomPoints_3I_5133_noH_v2_line3_pkl_v2_all_separate'
siteList = [t.split('/')[-1].split('.')[0] for t in glob.glob('%s/*.pickle' % data_path)]
siteList = ['%s/%s.pickle' % (data_path, t) for t in siteList]

X_train, X_eval, X_test = util.train_val_test_split_site(dataList=siteList, test_ratio=0.3)
county_n, train_n, eval_n, test_n = len(siteList), len(X_train), len(X_eval), len(X_test)
print('data total', len(siteList), 'train', len(X_train), 'eval', len(X_eval), 'test', len(X_test), flush=True)

# get the dataloader ready
scaler_inputs = pd.read_csv('ECOSOC_var_scalers.csv', index_col=0).loc[X_Features, :].to_numpy()
scaler_outputs = pd.read_csv('ECOSOC_output_scalers.csv', index_col=0).loc[y_Features, :].to_numpy()

batch_size = 256
length = len(pd.date_range('2000-01-01', '2020-12-31', freq='D').tolist())
train_ds = util.EcoSOC_dataset_pkl(X_train, X_Features=X_Features, y_Features=y_Features,
                                          scaler_inputs=scaler_inputs, scaler_outputs=scaler_outputs)
eval_ds = util.EcoSOC_dataset_pkl(X_eval, X_Features=X_Features, y_Features=y_Features,
                                         scaler_inputs=scaler_inputs, scaler_outputs=scaler_outputs)
test_ds = util.EcoSOC_dataset_pkl(X_test, X_Features=X_Features, y_Features=y_Features,
                                         scaler_inputs=scaler_inputs, scaler_outputs=scaler_outputs)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# load KGMLag model for predicting hidden layers
# ## define the network
input_dim = len(X_selectFeatures)
output_dim = [1,3,3,2]
hidden_dim = 64 # 64
mode = 'paraPheno_c2'
# model = net.GRUModel_hiera_4cell_add_noise_cropType_pheno(input_dim=input_dim,hidden_dim=hidden_dim,
#                               output_dim=output_dim)
model_hid = netSOC.GRUModel_hiera_4cell_add_noise_parameters_3layerDecoder_v2(input_dim=input_dim,hidden_dim=hidden_dim,
                              output_dim=output_dim,mode=mode)
model_hid = model_hid.to(device, non_blocking=True)

# reload model to make predictions
hidmodel_path='/panfs/jay/groups/15/jinzn/yang6956/SOC_KGML/train_DA/'
path_hid = '%s/train_2402/ag/model/gru-epoch50-batch256-240314_state_dict.pth'%(hidmodel_path)
checkpoint = torch.load(path_hid, map_location=torch.device('cpu'))
model_hid.load_state_dict(checkpoint)


years = 21
days = 365
n_x = len(X_Features)
n_y = len(y_Features)
n_total = n_x+n_y
countySamples = years * days

# set hyperparameters for SOC model training
n_hidden=64 #hidden state number
n_layer=2 #layer of GRU
dropout=0.2

#loss weights setup
compute_r2=R2Loss()
loss_fn = nn.MSELoss(reduction="mean")
loss_val_best = 500000
R2_best=0.25
starttime=time.time()
lr_adam= 1e-3

# select the SOC model for training: SOCGRU, SOCKGML, SOCKGML_res
model = netSOC.SOCKGML_hid16(n_x, hidden_dim, n_hidden, n_layer, dropout=0.2)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # every step_size, lr = lr*gamma

batch_total = int(train_n*0.9)
batch_size = 256  # this is the batch size for training
batch_n = (batch_total + batch_size-1) // batch_size
#during validation
train_losses = []
val_losses = []
n_epoch = 100
now = datetime.now().strftime('%y%m%d')
model_version='KGML_hid16_epoch{:d}_batch{:d}_{}'.format(n_epoch, batch_size, now)
basic_path='/panfs/jay/groups/15/jinzn/yang6956/SOC_KGML/train_DA/train_2402/SOC/'
path_save = basic_path+model_version
if not os.path.exists(path_save):
    os.makedirs(path_save, exist_ok=True)


# train model
model.train()
for epoch in range(n_epoch):
    train_loss,train_loss1,train_loss2=0.0,0.0,0.0
    val_loss,val_loss1,val_loss2=0.0,0.0,0.0
    loss, loss1, loss2 = 0.0,0.0,0.0

    for n, batch in enumerate(train_dl):
        optimizer.zero_grad()
        X_tmp, Y_tmp, Y_ann = batch

        X_tmp_ = X_tmp[~torch.any(torch.any(torch.isnan(torch.cat((X_tmp, Y_tmp), dim=2)), dim=2), dim=1), :, :]
        # Y_tmp_ = Y_tmp[~torch.any(torch.any(torch.isnan(torch.cat((X_tmp, Y_tmp), dim=2)), dim=2), dim=1), :, :]
        Y_train = Y_ann[~torch.any(torch.any(torch.isnan(torch.cat((X_tmp, Y_tmp), dim=2)), dim=2), dim=1), :, :]
        Y_train = Y_train.to(device)
        Y_train_prd = torch.zeros(Y_train.size(), device=device)
        hidden = model.init_hidden(X_tmp_.size(0))
        for i in range(years):
            X_ = X_tmp_[:, i*days:(i+1)*days, :].to(device)
            Y_ = Y_train[:, i:(i+1), :]
            optimizer.zero_grad()
            # calculate hidden layers from Qi's model
            hid_ind, hid_sum = model.init_hid(model_hid, X_)
            Y_prd, hidden = model(X_, hid_sum, hidden)
            hidden.detach_()

            loss1 = loss_fn(Y_prd[:,:,0], Y_[:,:,0])
            loss2 = loss_fn(Y_prd[:,:,1], Y_[:,:,1])
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss = train_loss + loss.item()
                train_loss1 = train_loss1 + loss1.item()
                train_loss2 = train_loss2 + loss2.item()
                Y_train_prd[:, i:(i + 1), :] = Y_prd[:, :, :]

        if n == 0:
            Y_train_total = Y_train.clone().detach()
            Y_train_prd_total = Y_train_prd.clone().detach()
        else:
            Y_train_total = torch.cat((Y_train_total, Y_train.clone().detach()), dim=0)
            Y_train_prd_total = torch.cat((Y_train_prd_total, Y_train_prd.clone().detach()), dim=0)

    scheduler.step()

    #validation
    model.eval()
    with torch.no_grad():
        train_loss = train_loss/(batch_n*years)
        train_loss1 = train_loss1/(batch_n*years)
        train_loss2 = train_loss2/(batch_n*years)
        train_losses.append([train_loss,train_loss1,train_loss2])
        # calculate r2
        train_R2_yld = compute_r2(Y_train_prd_total[:,:,0].squeeze(),Y_train_total[:,:,0].squeeze()).item()
        train_R2_soc = compute_r2(Y_train_prd_total[:,:,1].squeeze(),Y_train_total[:,:,1].squeeze()).item()
        train_R2 = [train_R2_yld, train_R2_soc]

        for n, batch_eval in enumerate(eval_dl):
            X_eval, Y_tmp, Y_eval = batch_eval
            X_eval_ = X_eval[~torch.any(torch.any(torch.isnan(torch.cat((X_eval, Y_tmp), dim=2)), dim=2), dim=1), :, :]
            Y_eval_ = Y_eval[~torch.any(torch.any(torch.isnan(torch.cat((X_eval, Y_tmp), dim=2)), dim=2), dim=1), :, :]
            Y_eval_ = Y_eval_.to(device)
            Y_eval_prd = torch.zeros(Y_eval_.size(),device=device)
            hidden = model.init_hidden(X_eval_.size(0))
            for i in range(years):
                X_ = X_eval_[:, i*days:(i+1)*days, :].to(device)
                hid_ind_val, hid_sum_val = model.init_hid(model_hid, X_)
                Y_prd, hidden = model(X_, hid_sum_val, hidden)
                hidden.detach_()
                Y_eval_prd[:, i:(i + 1), :] = Y_prd[:, :, :]

            if n == 0:
                Y_eval_total = Y_eval_.clone().detach()
                Y_eval_prd_total = Y_eval_prd.clone().detach()
            else:
                Y_eval_total = torch.cat((Y_eval_total, Y_eval_.clone().detach()), dim=0)
                Y_eval_prd_total = torch.cat((Y_eval_prd_total, Y_eval_prd.clone().detach()), dim=0)

        # calculate loss in evaluation
        val_loss1 = loss_fn(Y_eval_prd_total[:,:,0], Y_eval_total[:,:,0]).item()
        val_loss2 = loss_fn(Y_eval_prd_total[:,:,1], Y_eval_total[:,:,1]).item()
        val_loss = val_loss1 + val_loss2

        val_losses.append([val_loss,val_loss1,val_loss2])
        val_R2_yld = compute_r2(Y_eval_prd_total[:,:,0].squeeze(),Y_eval_total[:,:,0].squeeze()).item()
        val_R2_soc = compute_r2(Y_eval_prd_total[:,:,1].squeeze(),Y_eval_total[:,:,1].squeeze()).item()
        val_R2 = [val_R2_yld, val_R2_soc]
        # print(val_loss, loss_val_best, min(val_R2), R2_best, 'CHECK')
        if val_loss < loss_val_best and min(val_R2) > R2_best:
            loss_val_best=val_loss
            R2_best = min(val_R2)
            f0=open(path_save +'/log','w')
            f0.close()
            #os.remove(path_save)
            torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'R2': train_R2,
                    'loss': train_loss,
                    'los_val': val_loss,
                    'R2_val': val_R2,
                    }, path_save + '/log')
            model_best_state = model.state_dict()
        print("finished training epoch", epoch+1)
        mtime=time.time()
        print("train_loss: ", train_loss, "train_R2", train_R2,"val_loss:",val_loss,"val_R2", val_R2,\
              "loss val best:",loss_val_best,"R2 val best:",R2_best, f"Spending time: {mtime - starttime}s", flush=True)

        if min(train_R2) > 0.99:
            break
    model.train()

endtime = time.time()
path_fs = path_save + '/train'
torch.save({'train_losses': np.array(train_losses).transpose(),
            'val_losses': np.array(val_losses).transpose(),
            'model_state_dict': model.state_dict(),
            }, path_fs)
print("final train_loss:",train_loss,"final train_R2:",train_R2,"val_loss:",val_loss,"loss validation best:",loss_val_best, flush=True)
print(f"total Training time: {endtime - starttime}s", flush=True)

losses = {}
losses['train_losses'] = np.array(train_losses).transpose()
losses['val_losses'] = np.array(val_losses).transpose()

pd.DataFrame(train_losses).to_csv('%s/train_losses_log.csv'%(path_save))
pd.DataFrame(val_losses).to_csv('%s/val_losses_log.csv'%(path_save))


###
# test the model
model.load_state_dict(model_best_state)
model.eval()
with torch.no_grad():

    for n, batch in enumerate(test_dl):
        X_test, Y_tmp, Y_ann = batch
        X_test_ = X_test[~torch.any(torch.any(torch.isnan(torch.cat((X_test, Y_tmp), dim=2)), dim=2), dim=1), :, :]
        Y_ann_ = Y_ann[~torch.any(torch.any(torch.isnan(torch.cat((X_test, Y_tmp), dim=2)), dim=2), dim=1), :, :]
        Y_test = Y_ann_.to(device)
        Y_test_prd = torch.zeros(Y_ann_.size(), device=device)
        hidden = model.init_hidden(X_test_.size(0))

        for i in range(years):
            X_ = X_test_[:, i*days:(i+1)*days, :].to(device)
            hid_ind_test, hid_sum_test = model.init_hid(model_hid, X_)
            Y_prd, hidden = model(X_, hid_sum_test, hidden)
            hidden.detach_()
            Y_test_prd[:, i:(i + 1), :] = Y_prd[:, :, :]

        if n == 0:
            Y_test_total = Y_test.clone().detach()
            Y_test_prd_total = Y_test_prd.clone().detach()
        else:
            Y_test_total = torch.cat((Y_test_total, Y_test.clone().detach()), dim=0)
            Y_test_prd_total = torch.cat((Y_test_prd_total, Y_test_prd.clone().detach()), dim=0)

test_loss1 = loss_fn(Y_test_prd_total[:,:,0], Y_test_total[:,:,0]).item()
test_loss2 = loss_fn(Y_test_prd_total[:,:,1], Y_test_total[:,:,1]).item()

test_R2_yld = compute_r2(Y_test_prd_total[:,:,0].squeeze(),Y_test_total[:,:,0].squeeze()).item()
test_R2_soc = compute_r2(Y_test_prd_total[:,:,1].squeeze(),Y_test_total[:,:,1].squeeze()).item()
print('test loss yield', test_loss1, 'test loss SOC', test_loss2, 'test r2 yield', test_R2_yld, 'test r2 SOC', test_R2_soc, flush=True)

Ysim_retrive_train = []
Yobs_retrive_train = []
Ysim_retrive_test = []
Yobs_retrive_test = []
for i in range(n_y):
    # Ysim_retrive.append(Z_norm_reverse(Y_sim_all[:,:,i],scaler_output[i,:]).cpu().detach().numpy())
    # Yobs_retrive.append(Z_norm_reverse(Y_obs_all[:,:,i],scaler_output[i,:]).cpu().detach().numpy())
    Ysim_retrive_train.append(Z_norm_reverse(Y_train_prd_total[:,:,i],scaler_outputs[i,:]).cpu().detach().numpy())
    Yobs_retrive_train.append(Z_norm_reverse(Y_train_total[:,:,i],scaler_outputs[i,:]).cpu().detach().numpy())
    Ysim_retrive_test.append(Z_norm_reverse(Y_test_prd_total[:,:,i],scaler_outputs[i,:]).cpu().detach().numpy())
    Yobs_retrive_test.append(Z_norm_reverse(Y_test_total[:,:,i],scaler_outputs[i,:]).cpu().detach().numpy())


# plot the results: losses, scatters of all sites and time series by sites
outFolder = path_save + '/plots'
if not os.path.exists(outFolder):
    os.makedirs(outFolder, exist_ok=True)


# plot train and validation losses
loss_names=["Loss sum","Yield MSE","SOC MSE"]
util.plot_losses(losses,loss_names,outFolder=outFolder)

# plot
units = ['g C/$m^2$', 'g C/$m^2$']
vars = ['Yield', 'deltaSOC']
for i, var in enumerate(vars):
    unit = units[i]
    p = Ysim_retrive_train[i].flatten()
    o = Yobs_retrive_train[i].flatten()
    if var == 'deltaSOC':
        p = p * 1000
        o = o * 1000
    # print(i, max(o)-min(o), max(p)-min(p))
    util.plotScatterDense(x_=o, y_=p,outFolder=outFolder,saveFig=True,note='train_%s'%var,title=var + ' ' + unit)

for i, var in enumerate(vars):
    unit = units[i]
    p = Ysim_retrive_test[i].flatten()
    o = Yobs_retrive_test[i].flatten()
    if var == 'deltaSOC':
        p = p * 1000
        o = o * 1000

    data = pd.DataFrame()
    data['predictions'] = p
    data['synthetic'] = o
    data.to_csv('%s/test_series_%s.csv' % (outFolder, var))
    util.plotScatterDense(x_=o, y_=p,outFolder=outFolder,saveFig=True,note='test_%s'%var,title=var + ' ' + unit)


import scipy.stats
def lineargression(obs, sim):
    mask = ~np.isnan(obs)
    s1, inc1, r1, p1, se1 = scipy.stats.linregress(obs[mask], sim[mask])
    # RMSE = sqrt(mean_squared_error(obs, sim))
    MSE = np.square(np.subtract(obs[mask], sim[mask])).mean()
    RMSE = MSE ** 0.5
    return r1**2, RMSE

years = 21
timearr = np.arange(years) + 2000
units = ['g C/$m^2$', 'g C/$m^2$']
vars = ['Yield', 'deltaSOC']
ylims = [[0,500],[0,300], [0, 600], [-100,200]]
np.random.seed(0)
sites = np.random.randint(0, Ysim_retrive_test[0].shape[0], 20)
for site in sites:
    fig, ax = plt.subplots(len(y_Features), 1,figsize=(10, 5))
    plt.rcParams.update({'font.size': 16})
    for i, var in enumerate(vars):
        Ysim_t = Ysim_retrive_test[i][site,:].flatten()
        Yobs_t = Yobs_retrive_test[i][site,:].flatten()
        if var == 'deltaSOC':
            Ysim_t = Ysim_t * 1000
            Yobs_t = Yobs_t * 1000
        # print(Ysim[:5], Yobs[:5])
        ax[i].plot(timearr,Yobs_t,label="Synthetic, mean={:.2f}".format(np.mean(Yobs_t)),color='darkorange')
        ax[i].plot(timearr,Ysim_t, label="Predict, mean={:.2f}".format(np.mean(Ysim_t)),color='steelblue')
        ax[i].set_ylabel(units[i])
        ax[i].set_xticks(timearr[::2])
        R2_site, loss_site = lineargression(Ysim_t,Yobs_t)
        # loss_site=np.sqrt(torch.mean((Ysim_t - Yobs_t)**2))
        ax[i].set_title("site {:d} ".format(site) +var+\
                        " $R^2$="+str("{:.2f}".format(R2_site))+
                        " RMSE="+str("{:.2f}".format(loss_site)), fontsize=15)
        # fig.tight_layout()
        ax[i].grid()
        ax[i].set_ylim(ylims[i])
        ax[i].legend()
        fig.savefig('%s/timeseries_%s.png' % (outFolder, site), dpi=400)