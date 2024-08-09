import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from io import open
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from scipy.stats import gaussian_kde
import scipy.stats as stats
import subprocess as sp
from datetime import datetime
from sklearn.metrics import r2_score

device = "cuda" if torch.cuda.is_available() else "cpu"

def pearsonr2(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    r2_val = r_val*r_val
    return r2_val.to('cpu').numpy()
class R2Loss(nn.Module):
    #calculate coefficient of determination
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=False)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y


def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

#spin-up: bsz0 is number of year of data_sp you provided for spin up; bsz0=-1 means data_sp=[]
#data_sp is the data you provided
#return inihidden for simulation period with first year spin-uped
def spinup(model,data_sp,cycle,bsz):
    inihidden0=model.init_hidden(bsz)
    for c in range(cycle):
        output_dummy,inihidden0 = model(data_sp,inihidden0)
    return inihidden0
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
#for multi-task learning, sumloss
def myloss_mul_sum(output, target,loss_weights):
    loss = 0.0
    nout=output.size(2)
    for i in range(nout):
        loss = loss + loss_weights[i]*torch.mean((output[:,:,i] - target[:,:,i])**2)
    return loss
def scalar_maxmin(X):
    return (X - X.min())/(X.max() - X.min()),X.min(),X.max()

#generate input combine statini
#x should be size of [seq,batch,n_f1], statini be size of [1,batch,n_f2]
def load_ini(x,x_ini):
    nrep = x.size(0)
    x_ini=x_ini[0,:,:].view(1,x_ini.size(1),x_ini.size(2))
    return torch.cat((x,x_ini.repeat(nrep,1,1)),2)


def get_ini(x,ind,nout):
    initials=[]
    for i in range(len(ind)):
        initials.append(x[:,:,ind[i]].view(x.size(0),x.size(1),nout[i]))
    return initials

def Z_norm(X):
    X_mean=X.mean()
    X_std=np.std(np.array(X))
    return (X-X_mean)/X_std, X_mean, X_std

def Z_norm_reverse(X,Xscaler):
    return (X*Xscaler[1]+Xscaler[0])

#check whether start time is within the fertilized period
def dropout_check(start_t,fntime_ind):
    dropout_ind=False
    for t in fntime_ind:
        if start_t > t-10 and start_t < t+60:
            dropout_ind=True
    return dropout_ind

#sample data considering dropout and leadtime
def sample_data(X,Y,slw,slw05,totsq,fnfeature_ind):
    maxit=int((totsq-slw)/slw05+1)
    #find the fertilized time
    fntime_ind=np.where(X[:,1,fnfeature_ind].view(-1).to("cpu").numpy()>0)[0]
    #get sliding window data with dropout method
    for it in range(maxit):
        if it==0:
            X_new = X[slw05*it:slw05*it+slw,:,:]
            Y_new = Y[slw05*it:slw05*it+slw,:,:]
        else:
            if not dropout_check(slw05*it,fntime_ind):
                X_new = torch.cat((X_new,X[slw05*it:slw05*it+slw,:,:]),1)
                Y_new = torch.cat((Y_new,Y[slw05*it:slw05*it+slw,:,:]),1)
    #get focused data only for fertilized period with random leading time
    for t in fntime_ind:
        for b in range(X.size(1)):
            if t != fntime_ind[-1]:
                leadtime=np.random.randint(t-60,t-10)

                X_new = torch.cat((X_new,X[leadtime:leadtime+slw,b,:].view(slw,1,X.size(2))),1)
                Y_new = torch.cat((Y_new,Y[leadtime:leadtime+slw,b,:].view(slw,1,Y.size(2))),1)
    return X_new,Y_new

#sample data considering dropout and leadtime
def sample_data_FN(X,Y,totsq,fnfeature_ind):
    #find the fertilized time
    fntime_ind=np.where(X[:,1,fnfeature_ind].view(-1).to("cpu").numpy()>0)[0]
    #get focused data only for fertilized period with random leading time
    for t in fntime_ind:
        if t == fntime_ind[0]:
            X_new = X[t-30:t+90,:,:]
            Y_new = Y[t-30:t+90,:,:]
        else:
            X_new = torch.cat((X_new,X[t-30:t+90,:,:]),1)
            Y_new = torch.cat((Y_new,Y[t-30:t+90,:,:]),1)
    return X_new,Y_new

# GRU model
class SOCGRU(nn.Module):
    def __init__(self, n_inp, n_hidden, nlayers, dropout):
        super(SOCGRU, self).__init__()
        self.gru_basic = nn.GRU(n_inp, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        #attn for delta biomass prediction
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        # output = self.densor1(self.drop(output))
        # predict yield
        output1, hidden1 = self.gru_basic(inputs, hidden)
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs2 = torch.bmm(attn_weights,inputs2)
        yld = self.densor_yld(inputs2)
        # predict SOC
        # output2, hidden2 = self.gru_soc(torch.cat((yld.repeat(1,inputs.size(1),1),\
        #                                             self.drop(output1),inputs), 2), hidden[1])
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs3 = torch.bmm(attn_weights,inputs3)
        delta_soc = self.densor_soc(inputs3)
        outputs = torch.cat((yld,delta_soc),2)
        return outputs, hidden1
    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

class SOCGRU_cropyear(nn.Module):
    def __init__(self, n_inp, n_hidden, nlayers, dropout):
        super(SOCGRU, self).__init__()
        self.gru_basic = nn.GRU(n_inp, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        #attn for delta biomass prediction
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        # output = self.densor1(self.drop(output))
        # predict yield
        output1, hidden1 = self.gru_basic(inputs, hidden)
        output1, _ = pad_packed_sequence(output1, batch_first=True)
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs2 = torch.bmm(attn_weights,inputs2)
        yld = self.densor_yld(inputs2)
        # predict SOC
        # output2, hidden2 = self.gru_soc(torch.cat((yld.repeat(1,inputs.size(1),1),\
        #                                             self.drop(output1),inputs), 2), hidden[1])
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0),1,inputs.size(1))
        inputs3 = torch.bmm(attn_weights,inputs3)
        delta_soc = self.densor_soc(inputs3)
        outputs = torch.cat((yld,delta_soc),2)
        return outputs, hidden1
    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

class SOCGRU_res(nn.Module):
    def __init__(self, n_inp, n_hidden, nlayers, dropout):
        super(SOCGRU_res, self).__init__()
        self.gru_basic = nn.GRU(n_inp, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        # attn for residue
        self.attn_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        #attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden, res_previous):
        # output = self.densor1(self.drop(output))
        output1, hidden1 = self.gru_basic(inputs, hidden)
        # predict surface residue and soil residue
        inputs_res = self.drop(output1)
        attn_weights = F.softmax(self.attn_res(inputs_res), dim=1).view(inputs.size(0), 1, inputs.size(1))
        res = torch.bmm(attn_weights, inputs_res)
        res_ = self.densor_res(res)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        # print('res_size', res_.size(), inputs2.size())
        yld = self.densor_yld(torch.cat((inputs2, res_previous), dim=2))
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(torch.cat((inputs3, res_previous), dim=2))
        outputs = torch.cat(
            (res_[:, :, -2].view(res_.size(0), 1, 1), res_[:, :, -1].view(res_.size(0), 1, 1), yld, delta_soc), 2)
        return outputs, hidden1, res_

    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

class SOCGRU_res_cropyear(nn.Module):
    def __init__(self, n_inp, n_hidden, nlayers, dropout):
        super(SOCGRU_res, self).__init__()
        self.gru_basic = nn.GRU(n_inp, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        # attn for residue
        self.attn_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        #attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden, res_previous):
        # output = self.densor1(self.drop(output))
        output1, hidden1 = self.gru_basic(inputs, hidden)
        output1, _ = pad_packed_sequence(output1, batch_first=True)
        # predict surface residue and soil residue
        inputs_res = self.drop(output1)
        attn_weights = F.softmax(self.attn_res(inputs_res), dim=1).view(inputs.size(0), 1, inputs.size(1))
        res = torch.bmm(attn_weights, inputs_res)
        res_ = self.densor_res(res)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        # print('res_size', res_.size(), inputs2.size())
        yld = self.densor_yld(torch.cat((inputs2, res_previous), dim=2))
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(torch.cat((inputs3, res_previous), dim=2))
        outputs = torch.cat(
            (res_[:, :, -2].view(res_.size(0), 1, 1), res_[:, :, -1].view(res_.size(0), 1, 1), yld, delta_soc), 2)
        return outputs, hidden1, res_

    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

class SOCKGML_hid64(nn.Module):
    def __init__(self, n_inp, n_hid, n_hidden, nlayers, dropout):
        super(SOCKGML_hid64, self).__init__()
        self.gru_basic = nn.GRU(n_inp + n_hid, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        #attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hid, hidden):
        # output = self.densor1(self.drop(output))
        output1, hidden1 = self.gru_basic(torch.cat((inputs, hid), 2), hidden)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        yld = self.densor_yld(inputs2)
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(inputs3)
        outputs = torch.cat((yld, delta_soc), 2)
        return outputs, hidden1

    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum

class SOCKGML_hid64_cropyear(nn.Module):
    def __init__(self, n_inp, n_hid, n_hidden, nlayers, dropout):
        super(SOCKGML_hid64, self).__init__()
        self.gru_basic = nn.GRU(n_inp + n_hid, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        #attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hid, hidden):
        # output = self.densor1(self.drop(output))
        output1, hidden1 = self.gru_basic(torch.cat((inputs, hid), 2), hidden)
        output1, _ = pad_packed_sequence(output1, batch_first=True)
        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        yld = self.densor_yld(inputs2)
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(inputs3)
        outputs = torch.cat((yld, delta_soc), 2)
        return outputs, hidden1

    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum

class SOCKGML_hid16(nn.Module):
    def __init__(self, n_inp, n_hid, n_hidden, nlayers, dropout):
        super(SOCKGML_hid16, self).__init__()
        self.gru_basic = nn.GRU(n_inp + 16, n_hidden, 2, dropout=dropout, batch_first=True)
        # self.gru_soc = nn.GRU(ninp+nhid+1, nhid, 1, batch_first=True)
        # self.densor1 = nn.ReLU() #can test other function
        # self.densor2 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)

        #attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        #attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.densor_hid = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.ReLU=nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # may change to a small value
        for ii in range(4):
            self.attn_yld[ii*2].bias.data.zero_()
            self.attn_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii*2].bias.data.zero_()
            self.densor_yld[ii*2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii*2].bias.data.zero_()
            self.attn_soc[ii*2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii*2].bias.data.zero_()
            self.densor_soc[ii*2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hid, hidden):
        # output = self.densor1(self.drop(output))
        hid = self.densor_hid(hid)
        output1, hidden1 = self.gru_basic(torch.cat((inputs, hid), 2), hidden)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        yld = self.densor_yld(inputs2)
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(inputs3)
        outputs = torch.cat((yld, delta_soc), 2)
        return outputs, hidden1

    #bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum

class SOCKGML_res_hid64(nn.Module):
    def __init__(self, n_inp, n_hid, n_hidden, n_layer, dropout):
        super(SOCKGML_res_hid64, self).__init__()
        self.gru_basic = nn.GRU(n_inp + n_hid, n_hidden, n_layer, dropout=dropout, batch_first=True)
        # self.densor1 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.drop = nn.Dropout(dropout)

        # attn for predicting surface residue and soil residue
        self.attn_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1  # may change to a small value
        for ii in range(4):
            self.attn_yld[ii * 2].bias.data.zero_()
            self.attn_yld[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii * 2].bias.data.zero_()
            self.densor_yld[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii * 2].bias.data.zero_()
            self.attn_soc[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii * 2].bias.data.zero_()
            self.densor_soc[ii * 2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hid, hidden, res_previous):
        # output = self.densor1(self.drop(output))
        output1, hidden1 = self.gru_basic(torch.cat((inputs, hid), 2), hidden)

        # predict surface residue and soil residue
        inputs_res = self.drop(output1)
        attn_weights = F.softmax(self.attn_res(inputs_res), dim=1).view(inputs.size(0), 1, inputs.size(1))
        res = torch.bmm(attn_weights, inputs_res)
        res_ = self.densor_res(res)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        # print('res_size', res_.size(), inputs2.size())
        yld = self.densor_yld(torch.cat((inputs2, res_previous), dim=2))
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(torch.cat((inputs3, res_previous), dim=2))
        outputs = torch.cat(
            (res_[:, :, -2].view(res_.size(0), 1, 1), res_[:, :, -1].view(res_.size(0), 1, 1), yld, delta_soc), 2)
        return outputs, hidden1, res_

    # bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum

class SOCKGML_res_hid64_cropyear(nn.Module):
    def __init__(self, n_inp, n_hid, n_hidden, n_layer, dropout):
        super(SOCKGML_res_hid64, self).__init__()
        self.gru_basic = nn.GRU(n_inp + n_hid, n_hidden, n_layer, dropout=dropout, batch_first=True)
        # self.densor1 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.drop = nn.Dropout(dropout)

        # attn for predicting surface residue and soil residue
        self.attn_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ReLU = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1  # may change to a small value
        for ii in range(4):
            self.attn_yld[ii * 2].bias.data.zero_()
            self.attn_yld[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii * 2].bias.data.zero_()
            self.densor_yld[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii * 2].bias.data.zero_()
            self.attn_soc[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii * 2].bias.data.zero_()
            self.densor_soc[ii * 2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hid, hidden, res_previous):
        # output = self.densor1(self.drop(output))
        output1, hidden1 = self.gru_basic(torch.cat((inputs, hid), 2), hidden)
        output1, _ = pad_packed_sequence(output1, batch_first=True)
        # predict surface residue and soil residue
        inputs_res = self.drop(output1)
        attn_weights = F.softmax(self.attn_res(inputs_res), dim=1).view(inputs.size(0), 1, inputs.size(1))
        res = torch.bmm(attn_weights, inputs_res)
        res_ = self.densor_res(res)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        # print('res_size', res_.size(), inputs2.size())
        yld = self.densor_yld(torch.cat((inputs2, res_previous), dim=2))
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(torch.cat((inputs3, res_previous), dim=2))
        outputs = torch.cat(
            (res_[:, :, -2].view(res_.size(0), 1, 1), res_[:, :, -1].view(res_.size(0), 1, 1), yld, delta_soc), 2)
        return outputs, hidden1, res_, inputs3

    # bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum

class SOCKGML_res_hid16(nn.Module):
    def __init__(self, n_inp, n_hid, n_hidden, n_layer, dropout):
        super(SOCKGML_res_hid16, self).__init__()
        self.gru_basic = nn.GRU(n_inp + 16, n_hidden, n_layer, dropout=dropout, batch_first=True)
        # self.densor1 = nn.Linear(nhid, nout)
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.drop = nn.Dropout(dropout)

        # attn for predicting surface residue and soil residue
        self.attn_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_res = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # attn for yield
        self.attn_yld = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_yld = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # attn for SOC prediction
        self.attn_soc = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.densor_soc = nn.Sequential(
            nn.Linear(n_hidden + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.densor_hid = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.ReLU = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1  # may change to a small value
        for ii in range(4):
            self.attn_yld[ii * 2].bias.data.zero_()
            self.attn_yld[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.densor_yld[ii * 2].bias.data.zero_()
            self.densor_yld[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.attn_soc[ii * 2].bias.data.zero_()
            self.attn_soc[ii * 2].weight.data.uniform_(-initrange, initrange)
            self.densor_soc[ii * 2].bias.data.zero_()
            self.densor_soc[ii * 2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hid, hidden, res_previous):
        # output = self.densor1(self.drop(output))
        hid = self.densor_hid(hid)
        output1, hidden1 = self.gru_basic(torch.cat((inputs, hid), 2), hidden)

        # predict surface residue and soil residue
        inputs_res = self.drop(output1)
        attn_weights = F.softmax(self.attn_res(inputs_res), dim=1).view(inputs.size(0), 1, inputs.size(1))
        res = torch.bmm(attn_weights, inputs_res)
        res_ = self.densor_res(res)

        # predict yield
        inputs2 = self.drop(output1)
        attn_weights = F.softmax(self.attn_yld(inputs2), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs2 = torch.bmm(attn_weights, inputs2)
        # print('res_size', res_.size(), inputs2.size())
        yld = self.densor_yld(torch.cat((inputs2, res_previous), dim=2))
        # predict SOC
        inputs3 = self.drop(output1)
        attn_weights = F.softmax(self.attn_soc(inputs3), dim=1).view(inputs.size(0), 1, inputs.size(1))
        inputs3 = torch.bmm(attn_weights, inputs3)
        delta_soc = self.densor_soc(torch.cat((inputs3, res_previous), dim=2))
        outputs = torch.cat(
            (res_[:, :, -2].view(res_.size(0), 1, 1), res_[:, :, -1].view(res_.size(0), 1, 1), yld, delta_soc), 2)
        return outputs, hidden1, res_, inputs3

    # bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum



class SOCKGML_res_hid16_finetune(nn.Module):
    def __init__(self, n_hidden):
        super(SOCKGML_res_hid16_finetune, self).__init__()

        self.densor_tune = nn.Sequential(
            nn.Linear(n_hidden + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.ReLU = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1  # may change to a small value
        for ii in range(1):
            self.densor_tune[ii * 2].bias.data.zero_()
            self.densor_tune[ii * 2].weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, obs_depth, delta_soc_pred):
        delta_soc_tuned = self.densor_tune(torch.cat((inputs, obs_depth, delta_soc_pred), dim=2))
        return delta_soc_tuned

    # bsz should be batch size
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(2, bsz, self.n_hidden)

    # load hidden states from Qi's model
    def init_hid(self, model_hid, hid_inputs):
        model_hid.eval()
        with torch.no_grad():
            Y_pred, _, hid = model_hid(hid_inputs)
            # Y_pred_retrive = Y_pred.cpu().detach().numpy()
            hid_1, hid_2, hid_3, hid_4 = hid
            hid_ind = [hid_1, hid_2, hid_3, hid_4]
            hid_sum = hid_1 + hid_2 + hid_3 + hid_4

        return hid_ind, hid_sum


class GRUModel_hiera_4cell_add_noise_parameters_3layerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=[1, 3, 3, 2], noise_cv=0.01, mode='para'):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cv = noise_cv
        self.mode = mode

        # GRU layers
        self.gruCell_1 = nn.GRUCell(input_size=self.input_dim, hidden_size=hidden_dim)
        self.gruCell_2 = nn.GRUCell(input_size=self.input_dim + 1, hidden_size=hidden_dim)
        self.gruCell_3 = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)
        self.gruCell_4 = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)

        # Fully connected layer
        self.outDim = output_dim
        self.fc_1 = nn.Linear(hidden_dim, self.outDim[0])
        self.fc_2 = nn.Linear(hidden_dim, self.outDim[1])
        self.fc_3 = nn.Linear(hidden_dim, self.outDim[2])
        self.fc_4 = nn.Linear(hidden_dim, self.outDim[3])

        if self.mode == 'para':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + len(self.parasList), 32)  # + paras
            self.fc_2_v1 = nn.Linear(self.outDim[1] + len(self.parasList), 32)  # + paras
            self.fc_3_v1 = nn.Linear(self.outDim[2] + len(self.parasList), 32)  # + paras
            self.fc_4_v1 = nn.Linear(self.outDim[3] + len(self.parasList), 32)  # + paras

        elif self.mode == 'basic':
            self.fc_1_v1 = nn.Linear(self.outDim[0], 32)  #
            self.fc_2_v1 = nn.Linear(self.outDim[1], 32)  #
            self.fc_3_v1 = nn.Linear(self.outDim[2], 32)  #
            self.fc_4_v1 = nn.Linear(self.outDim[3], 32)  #

        elif self.mode == 'crop':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + 1, 32)  # + croptype
            self.fc_2_v1 = nn.Linear(self.outDim[1] + 1, 32)  # + croptype
            self.fc_3_v1 = nn.Linear(self.outDim[2] + 1, 32)  # + croptype
            self.fc_4_v1 = nn.Linear(self.outDim[3] + 1, 32)  # + croptype

        elif self.mode == 'paraPheno_c2':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + len(self.parasList), 32)  # + paras
            self.fc_2_v1 = nn.Linear(self.outDim[1] + len(self.parasList) + 1, 32)  # + paras
            self.fc_3_v1 = nn.Linear(self.outDim[2] + len(self.parasList), 32)  # + paras
            self.fc_4_v1 = nn.Linear(self.outDim[3] + len(self.parasList), 32)  # + paras

        self.fc_1_v2 = nn.Linear(32, 32)
        self.fc_2_v2 = nn.Linear(32, 32)
        self.fc_3_v2 = nn.Linear(32, 32)
        self.fc_4_v2 = nn.Linear(32, 32)

        self.fc_1_v3 = nn.Linear(32, hidden_dim)
        self.fc_2_v3 = nn.Linear(32, hidden_dim)
        self.fc_3_v3 = nn.Linear(32, hidden_dim)
        self.fc_4_v3 = nn.Linear(32, hidden_dim)

        self.relu = nn.ReLU()

        # Bn of inputs
        self.bn = nn.BatchNorm1d(input_dim)


    def forward(self, x, hidden=None, initLAI=None, isTrain=False, updateState=None, seq_lengthList=None):
        self.batchsize = x.size(0)

        # initialize inputs
        if hidden == None:
            # Initializing hidden state for first input with zeros
            self.hidden = []
            for i in range(4):
                h0 = torch.zeros(x.size(0), self.hidden_dim).requires_grad_().to(device)
                self.hidden.append(h0.detach())
        else:
            self.hidden = [h.detach() for h in hidden]
        if initLAI == None:
            self.LAI_previous = torch.zeros(x.size(0), 1).to(device).detach()
        else:
            self.LAI_previous = initLAI.detach()

        # deploy BN normalization
        # x_debug = torch.clone(x.detach())
        # x_debug_n = x_debug.detach().cpu().numpy()
        # x = x.permute(0, 2, 1)
        # x = self.bn(x)
        # x = x.permute(0, 2, 1)
        # x_debug_n_bn = x.detach().cpu().numpy()
        self.hidden_1, self.hidden_2, self.hidden_3, self.hidden_4 = self.hidden

        # Forward propagation by passing in the input and hidden state into the model
        # loop for sequence length
        # RNNcell cannot receive packed sequence, so we didn't pack them, it will take more computing but it is ok
        outList1 = []
        outList2 = []
        outList3 = []
        outList4 = []
        hList1 = []
        hList2 = []
        hList3 = []
        hList4 = []

        for i in range(x.shape[1]):
            x_forward = x[:, i, :]
            self.hidden_1 = self.gruCell_1(x_forward, self.hidden_1)
            self.hidden_2 = self.gruCell_2(torch.cat([x_forward, self.LAI_previous], dim=1), self.hidden_2)
            cell1_out = self.fc_1(self.hidden_1)
            cell2_out = self.fc_2(self.hidden_2)

            self.hidden_3 = self.gruCell_3(torch.cat([self.hidden_1.detach(), self.hidden_2.detach()], dim=1),
                                           self.hidden_3)
            cell3_out = self.fc_3(self.hidden_3)

            self.hidden_4 = self.gruCell_4(torch.cat([self.hidden_1.detach(), self.hidden_3.detach()], dim=1),
                                           self.hidden_4)
            cell4_out = self.fc_4(self.hidden_4)
            self.LAI_previous = cell4_out[:, 0].view([-1, 1]).detach()

            outList1.append(cell1_out)
            outList2.append(cell2_out)
            outList3.append(cell3_out)
            outList4.append(cell4_out)

            hList1.append(self.hidden_1)
            hList2.append(self.hidden_2)
            hList3.append(self.hidden_3)
            hList4.append(self.hidden_4)

        out1 = torch.stack(outList1)
        out2 = torch.stack(outList2)
        out3 = torch.stack(outList3)
        out4 = torch.stack(outList4)

        out = torch.cat([out1, out2, out3, out4], dim=2)

        hTensor1 = torch.stack(hList1)
        hTensor2 = torch.stack(hList2)
        hTensor3 = torch.stack(hList3)
        hTensor4 = torch.stack(hList4)

        return out, [self.hidden_1, self.hidden_2, self.hidden_3, self.hidden_3], \
               [hTensor1, hTensor2, hTensor3, hTensor4]


class GRUModel_hiera_4cell_add_noise_parameters_3layerDecoder_v2(nn.Module):
    '''
     v2: speed up the network runs
    '''

    def __init__(self, input_dim, hidden_dim, output_dim=[1, 3, 3, 2], noise_cv=0.01, mode='para'):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cv = noise_cv
        self.mode = mode
        self.parasList = ['CropType', 'VCMX', 'CHL4', 'GROUPX', 'STMX', 'GFILL', 'SLA1', 'Bulk density',
                          'Field capacity'
            , 'Wilting point', 'Ks', 'Sand content', 'Silt content', 'SOC', 'Fertilizer']

        # GRU layers
        self.gruCell_1 = nn.GRUCell(input_size=self.input_dim, hidden_size=hidden_dim)
        self.gruCell_2 = nn.GRUCell(input_size=self.input_dim + 1, hidden_size=hidden_dim)
        self.gruCell_3 = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)
        self.gruCell_4 = nn.GRUCell(input_size=hidden_dim * 2, hidden_size=hidden_dim)

        # Fully connected layer
        self.outDim = output_dim
        self.fc_1 = nn.Linear(hidden_dim, self.outDim[0])
        self.fc_2 = nn.Linear(hidden_dim, self.outDim[1])
        self.fc_3 = nn.Linear(hidden_dim, self.outDim[2])
        self.fc_4 = nn.Linear(hidden_dim, self.outDim[3])

        if self.mode == 'para':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + len(self.parasList), 32)  # + paras
            self.fc_2_v1 = nn.Linear(self.outDim[1] + len(self.parasList), 32)  # + paras
            self.fc_3_v1 = nn.Linear(self.outDim[2] + len(self.parasList), 32)  # + paras
            self.fc_4_v1 = nn.Linear(self.outDim[3] + len(self.parasList), 32)  # + paras

        elif self.mode == 'basic':
            self.fc_1_v1 = nn.Linear(self.outDim[0], 32)  #
            self.fc_2_v1 = nn.Linear(self.outDim[1], 32)  #
            self.fc_3_v1 = nn.Linear(self.outDim[2], 32)  #
            self.fc_4_v1 = nn.Linear(self.outDim[3], 32)  #

        elif self.mode == 'crop':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + 1, 32)  # + croptype
            self.fc_2_v1 = nn.Linear(self.outDim[1] + 1, 32)  # + croptype
            self.fc_3_v1 = nn.Linear(self.outDim[2] + 1, 32)  # + croptype
            self.fc_4_v1 = nn.Linear(self.outDim[3] + 1, 32)  # + croptype

        elif self.mode == 'paraPheno_c2':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + len(self.parasList), 32)  # + paras
            self.fc_2_v1 = nn.Linear(self.outDim[1] + len(self.parasList) + 1, 32)  # + paras
            self.fc_3_v1 = nn.Linear(self.outDim[2] + len(self.parasList), 32)  # + paras
            self.fc_4_v1 = nn.Linear(self.outDim[3] + len(self.parasList), 32)  # + paras

        elif self.mode == 'cropPheno_c2':
            self.fc_1_v1 = nn.Linear(self.outDim[0] + 1, 32)  # + croptype
            self.fc_2_v1 = nn.Linear(self.outDim[1] + 2, 32)  # + croptype & pheno
            self.fc_3_v1 = nn.Linear(self.outDim[2] + 1, 32)  # + croptype
            self.fc_4_v1 = nn.Linear(self.outDim[3] + 1, 32)  # + croptype

        self.fc_1_v2 = nn.Linear(32, 32)
        self.fc_2_v2 = nn.Linear(32, 32)
        self.fc_3_v2 = nn.Linear(32, 32)
        self.fc_4_v2 = nn.Linear(32, 32)

        self.fc_1_v3 = nn.Linear(32, hidden_dim)
        self.fc_2_v3 = nn.Linear(32, hidden_dim)
        self.fc_3_v3 = nn.Linear(32, hidden_dim)
        self.fc_4_v3 = nn.Linear(32, hidden_dim)

        self.relu = nn.ReLU()

        # Bn of inputs
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x, hidden=None, initLAI=None, isTrain=True, updateState=None, seq_lengthList=None):
        self.batchsize = x.size(0)

        # initialize inputs
        if hidden == None:
            # Initializing hidden state for first input with zeros
            self.hidden = []
            for i in range(4):
                h0 = torch.zeros(x.size(0), self.hidden_dim).requires_grad_().to(device)
                self.hidden.append(h0.detach())
        else:
            self.hidden = [h.detach() for h in hidden]
        if initLAI == None:
            self.LAI_previous = torch.zeros(x.size(0), 1).to(device).detach()
        else:
            self.LAI_previous = initLAI.detach()

        # deploy BN
        # x_debug = torch.clone(x.detach())
        # x_debug_n = x_debug.detach().cpu().numpy()
        # x = x.permute(0, 2, 1)
        # x = self.bn(x)
        # x = x.permute(0, 2, 1)
        # x_debug_n_bn = x.detach().cpu().numpy()
        self.hidden_1, self.hidden_2, self.hidden_3, self.hidden_4 = self.hidden

        # Forward propagation by passing in the input and hidden state into the model
        # loop for sequence length
        # RNNcell cannot receive packed sequence, so we didn't pack them, it will take more comperting but it is ok
        out1 = torch.zeros((self.batchsize, x.shape[1], self.outDim[0])).to(device)
        out2 = torch.zeros((self.batchsize, x.shape[1], self.outDim[1])).to(device)
        out3 = torch.zeros((self.batchsize, x.shape[1], self.outDim[2])).to(device)
        out4 = torch.zeros((self.batchsize, x.shape[1], self.outDim[3])).to(device)
        hTensor1 = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor2 = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor3 = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor4 = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor1_v = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor2_v = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor3_v = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)
        hTensor4_v = torch.zeros((self.batchsize, x.shape[1], self.hidden_dim)).to(device)

        for i in range(x.shape[1]):

            x_forward = x[:, i, :]
            self.hidden_1 = self.gruCell_1(x_forward, self.hidden_1)
            self.hidden_2 = self.gruCell_2(torch.cat([x_forward, self.LAI_previous], dim=1), self.hidden_2)
            cell1_out = self.fc_1(self.hidden_1)
            cell2_out = self.fc_2(self.hidden_2)

            self.hidden_3 = self.gruCell_3(torch.cat([self.hidden_1.detach(), self.hidden_2.detach()], dim=1),
                                           self.hidden_3)
            cell3_out = self.fc_3(self.hidden_3)

            self.hidden_4 = self.gruCell_4(torch.cat([self.hidden_1.detach(), self.hidden_3.detach()], dim=1),
                                           self.hidden_4)
            cell4_out = self.fc_4(self.hidden_4)
            self.LAI_previous = cell4_out[:, 0].view([-1, 1]).detach()

            out1[:, i, :] = cell1_out
            out2[:, i, :] = cell2_out
            out3[:, i, :] = cell3_out
            out4[:, i, :] = cell4_out

            hTensor1[:, i, :] = self.hidden_1
            hTensor2[:, i, :] = self.hidden_2
            hTensor3[:, i, :] = self.hidden_3
            hTensor4[:, i, :] = self.hidden_4


        out = torch.cat([out1, out2, out3, out4], dim=2)

        if isTrain:
            return out, [self.hidden_1, self.hidden_2, self.hidden_3, self.hidden_4], \
                   [hTensor1, hTensor2, hTensor3, hTensor4]
        else:
            return out, [self.hidden_1, self.hidden_2, self.hidden_3, self.hidden_4]