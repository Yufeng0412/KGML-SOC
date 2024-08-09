import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import os
import pandas as pd
import glob
import time
from scipy import stats
from sklearn.metrics import r2_score


version = int(sys.version.split()[0].split('.')[1])
if version > 7:
    import pickle
else:
    import pickle5 as pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


def listSplit(l,ref_index):
    inList = []
    outList = []
    for i,t in enumerate(l):
        if i in ref_index:
            inList.append(t)
        else:
            outList.append(t)
    return inList,outList

def train_val_test_split_site(dataList, test_ratio=0.1):
    np.random.seed(0)
    val_ratio = test_ratio / (1 - test_ratio)
    indexList = np.arange(0, len(dataList))
    np.random.shuffle(indexList)

    # split train and test with no leak strategy
    indexTest = indexList[:int(test_ratio * len(dataList))]
    test, train_t = listSplit(dataList, indexTest)
    # y_test, y_train_t = listSplit(y, indexTest)

    # split train and vali
    indextrain = np.arange(0, len(train_t))
    np.random.shuffle(indextrain)
    indexVali = indextrain[:int(val_ratio * len(indextrain))]
    train, val = listSplit(train_t, indexVali)
    # y_train, y_val = listSplit(y_train_t, indexVali)

    return train, val, test


class EcoNet_dataset_pkl_yearly(Dataset):
    def __init__(self, data_X_pathes, X_selectFeatures=None, y_selectFeatures=None, y_NormCoef=None, length=7671):
        self.data_X_pathes = data_X_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        self.length = length

    def __getitem__(self, index, batchmodel=True):

        if batchmodel:

            self.data = self.load_object(self.data_X_pathes[index])
            self.data = self.data[self.data['DOY'] != 366].reset_index(drop=True)

            # self.data_X = self.load_object(self.data_X_pathes[index])
            # self.data_y = self.load_object(self.data_y_pathes[index])
            mc_mask = (self.data['Growseason'] > 0).astype(int).to_numpy().reshape((self.data.shape[0], 1))
            cc_mask = (self.data['Growseason'] == 0).astype(int).to_numpy().reshape((self.data.shape[0], 1))

            out_X = np.array(self.data[self.X_selectFeatures]).astype(np.float32)
            out_y = np.array(self.data[self.y_selectFeatures]).astype(np.float32)

            if not self.y_NormCoef == None:
                for i, t in enumerate(self.y_NormCoef):
                    out_y[:, i] = out_y[:, i] * t

            return out_X, out_y, mc_mask, cc_mask
        else:
            index_s, index_d = index
            self.data_X = self.load_object(self.data_X_pathes[index_s])
            self.data_y = self.load_object(self.data_y_pathes[index_s])

            if self.X_selectFeatures == None:
                out_X = self.data_X.iloc[index_d].tolist()
            else:
                out_X = self.data_X[self.X_selectFeatures].iloc[index_d].tolist()
            if self.y_selectFeatures == None:
                out_y = self.data_y.iloc[index_d].tolist()
            else:
                out_y = self.data_y[self.y_selectFeatures].iloc[index_d].tolist()

            if not self.y_NormCoef == None:
                for i, t in enumerate(self.y_NormCoef):
                    out_y[i] = out_y[i] * t

                    return out_X, out_y

    def load_object(self, filename):
        # data = pd.read_csv(filename)
        # return data
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_X_pathes)


class EcoNet_dataset_pkl_yearly_cropyear_365(Dataset):
    def __init__(self, data_X_pathes, X_selectFeatures=None, y_selectFeatures=None, y_NormCoef=None, length=7671):
        self.data_X_pathes = data_X_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        self.length = length

    def __getitem__(self, index, batchmodel=True):

        self.data = self.load_object(self.data_X_pathes[index])

        out_X = np.zeros(([365 * 20, len(self.X_selectFeatures)])).astype(np.float32)
        out_y = np.zeros(([365 * 20, len(self.y_selectFeatures)])).astype(np.float32)
        days_mask = np.zeros(([20 * 365, 1]))
        mc_mask = np.zeros(([20 * 365, 1]))
        cc_mask = np.zeros(([20 * 365, 1]))
        for i, year in enumerate(range(2000, 2020)):
            gs_start = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year))].index
            gs_end = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year + 1))].index
            # print(year, gs_start, gs_end)
            if len(gs_start) > 0 and len(gs_end) > 0:
                crop_year = self.data.loc[gs_start[0]:gs_end[0], :]
                mc_mask[gs_start[0]: (gs_start[-1] + 1), :] = 1
                cc_mask[(gs_start[-1] + 1): gs_end[0], :] = 1
            elif len(gs_start) > 0:
                crop_year = self.data.loc[gs_start[0]:(gs_start[0] + 365), :]
                mc_mask[gs_start[0]: (gs_start[-1] + 1), :] = 1
                cc_mask[(gs_start[-1] + 1): (gs_start[0] + 365), :] = 1
            else:
                continue
            days = crop_year.shape[0]
            if days >= 365:
                out_X[365 * i:365 * (i + 1), :] = crop_year[self.X_selectFeatures].to_numpy()[:365, :]
                out_y[365 * i:365 * (i + 1), :] = crop_year[self.y_selectFeatures].to_numpy()[:365, :]
                days_mask[365 * i:365 * (i + 1), :] = 1

            else:
                out_X[365 * i:(365 * i + days), :] = crop_year[self.X_selectFeatures].to_numpy()
                out_y[365 * i:(365 * i + days), :] = crop_year[self.y_selectFeatures].to_numpy()
                days_mask[365 * i:(365 * i + days), :] = 1

        if not self.y_NormCoef == None:
            for i, t in enumerate(self.y_NormCoef):
                out_y[:, i] = out_y[:, i] * t

        return out_X, out_y, days_mask, mc_mask, cc_mask


    def load_object(self, filename):
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_X_pathes)


class EcoNet_dataset_pkl_yearly_cropyear(Dataset):
    def __init__(self, data_X_pathes, X_selectFeatures=None, y_selectFeatures=None, y_NormCoef=None, length=7671):
        self.data_X_pathes = data_X_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_selectFeatures = X_selectFeatures
        self.y_selectFeatures = y_selectFeatures
        self.y_NormCoef = y_NormCoef
        self.length = length

    def __getitem__(self, index, batchmodel=True):

        self.data = self.load_object(self.data_X_pathes[index])

        max_gslength = 390 # 365 d add with 25 d planting disturbance
        years = 20
        out_X = np.zeros(([max_gslength * years, len(self.X_Features)])).astype(np.float32)
        out_y = np.zeros(([max_gslength * years, len(self.y_Features)])).astype(np.float32)
        seq_length = np.zeros(([years])).astype(np.int)
        ann_y = np.zeros([years, len(self.y_Features)], dtype=np.float32)
        days_mask = np.zeros(([years * max_gslength, 1]))
        mc_mask = np.zeros(([years * max_gslength, 1]))
        cc_mask = np.zeros(([years * max_gslength, 1]))

        for i, year in enumerate(range(2000, 2020)):
            gs_start = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year))].index
            gs_end = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year + 1))].index

            if len(gs_start) > 0 and len(gs_end) > 0:
                crop_year = self.data.loc[gs_start[0]:gs_end[0]-1, :]
                mc_mask[gs_start[0]: (gs_start[-1] + 1), :] = 1
                cc_mask[(gs_start[-1] + 1): gs_end[0], :] = 1
            elif len(gs_start) > 0:
                crop_year = self.data.loc[gs_start[0]:(gs_start[0] + max_gslength-1), :]
                mc_mask[gs_start[0]: (gs_start[-1] + 1), :] = 1
                cc_mask[(gs_start[-1] + 1): (gs_start[0] + max_gslength), :] = 1
            else:
                continue
            days = crop_year.shape[0]
            seq_length[i] = days
            out_X[max_gslength * i:(max_gslength * i + days), :] = crop_year[self.X_selectFeatures].to_numpy()
            out_y[max_gslength * i:(max_gslength * i + days), :] = crop_year[self.y_selectFeatures].to_numpy()
            days_mask[max_gslength * i:(max_gslength * i + days), :] = 1


        if not self.y_NormCoef == None:
            for i, t in enumerate(self.y_NormCoef):
                out_y[:, i] = out_y[:, i] * t

        return out_X, out_y, days_mask, mc_mask, cc_mask


    def load_object(self, filename):
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_X_pathes)


class Optimization_decoder:
    def __init__(self, model, loss_fn=None, optimizer=None, exp_lr_scheduler=None, hiera=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.exp_lr_scheduler = exp_lr_scheduler
        self.train_losses = []
        self.val_losses = []
        self.train_losses_main = []
        self.val_losses_main = []
        self.train_losses_de = []
        self.val_losses_de = []
        self.hiera = hiera

    def changePadValue(self, y, seq_length):
        # replace the padding to -999
        t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                    batch_first=True, enforce_sorted=False)
        y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True, padding_value=-999)
        return y_repad.detach()

    def train_step(self, x, y, mc_mask, cc_mask, hidden=None):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        if self.hiera:
            yhat, hidden, hTensor = self.model(x, hidden=hidden, seq_lengthList=mc_mask, isTrain=True)
        else:
            yhat, hTensor = self.model(x, hidden=hidden, seq_lengthList=mc_mask, isTrain=True)
        # print(x.size(), y.size(), yhat.size())
        # replace the padding to -999
        # y_repad = self.changePadValue(y=y,seq_length=x_length)

        # calculate the mask
        # loss_main = 0
        # for i in range(y.shape[-1]):
        #     y_repad_i = y_repad[:,:,i].detach()
        #     mask = (y_repad_i != -999).float()

        #     # Computes MSEloss
        #     yhat_i = yhat[:,:,i]
        #     y_i = y[:,:,i]
        #     loss_main += torch.sum(((yhat_i-y_i)*mask)**2) / torch.sum(mask)

        # mask = (y_repad.detach() != -999).float()
        # mask out main crop and cover crop windows and calcualte the loss seperately

        # mc_mask = days_mask.detach().clone()
        # mc_mask[:,185:,:] = 0
        # cc_mask = days_mask.detach().clone()
        # cc_mask[:,:185,:] = 0
        loss_mc = torch.sum(((yhat - y) * mc_mask) ** 2) / torch.sum(mc_mask)
        # loss_mc.backward(retain_graph=True)
        # self.optimizer.step()
        # # self.optimizer.zero_grad()

        loss_cc = torch.sum(((yhat - y) * cc_mask) ** 2) / torch.sum(cc_mask)
        loss = loss_mc + loss_cc
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # calculate reconstruction loss
        # loss_de = 0
        # for h,h_v in zip(hTensor,hTensor_v):
        #     if self.hiera:
        #         h_repad = self.changePadValue(y=h,seq_length=x_length)
        #         mask = (h_repad != -999).float()
        #
        #         loss_de += torch.sum(((h_v-h)*mask)**2) / torch.sum(mask)
        #     else:
        #         loss_de += torch.sum(((h_v-h))**2) / torch.sum(h)

        # loss = loss_main
        # Computes gradients

        # Updates parameters and zeroes gradients
        # self.optimizer.step()
        # self.optimizer.zero_grad()

        # Returns the loss
        return loss.item(), hidden

    def vali_step(self, x_val, y_val, mc_mask, cc_mask, hidden=None):
        self.model.eval()
        yhat, hidden, hTensor = self.model(x_val, seq_lengthList=mc_mask, isTrain=True)
        # replace the padding to -999
        # y_repad = self.changePadValue(y=y_val,seq_length=x_val_length)

        # calculate the mask
        val_loss_main = 0
        # for i in range(y_val.shape[-1]):
        #     # y_repad_i = y_repad[:,:,i].detach()
        #     # mask = (y_repad_i != -999).float()
        #
        #     # Computes MSEloss
        #     yhat_i = yhat[:,:,i]
        #     y_val_i = y_val[:,:,i]
        #     mask_i = days_mask[:,:,0]
        #     val_loss_main += torch.sum(((yhat_i-y_val_i)*mask_i)**2) / torch.sum(mask_i)

        # mc_mask = days_mask.detach().clone()
        # mc_mask[:, 185:, :] = 0
        # cc_mask = days_mask.detach().clone()
        # cc_mask[:, :185, :] = 0
        loss_mc_val = torch.sum(((yhat - y_val) * mc_mask) ** 2) / torch.sum(mc_mask)
        loss_cc_val = torch.sum(((yhat - y_val) * cc_mask) ** 2) / torch.sum(cc_mask)
        val_loss = loss_mc_val + loss_cc_val
        #
        # # calculate reconstruction loss
        # val_loss_de = 0
        # for h,h_v in zip(hTensor,hTensor_v):
        #     if self.hiera:
        #         h_repad = self.changePadValue(y=h,seq_length=x_val_length)
        #         mask = (h_repad != -999).float()
        #
        #         val_loss_de += torch.sum(((h_v-h)*mask)**2) / torch.sum(mask)
        #     else:
        #         val_loss_de += torch.sum(((h_v-h))**2) / torch.sum(h)

        # val_loss = val_loss_main
        return val_loss.item(), hidden

    def train(self, train_loader, val_loader, n_epochs=50, n_features=1, interv=50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []

            train_loader.epochStart()
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            while True:

                batch = train_loader.getbatch()
                if batch == False:
                    break
                x_batch, y_batch, x_length = batch
                if timeCount:
                    t_d = time.time()
                    dif_t = t_d - tt
                    tt = t_d
                    print('load batch taking %.3f s' % dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length)
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)
                if timeCount:
                    t_d = time.time()
                    dif_t = t_d - tt
                    tt = t_d
                    print('train taking %.3f s' % dif_t)
                bN += 1
                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t = t_d - t_s
                        t_s = t_d
                        print('Taking %.3f s' % dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{train_loader.epoch_batches}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)

            # lr decay
            self.exp_lr_scheduler.step()

            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                val_loader.epochStart()
                while True:
                    batch_v = val_loader.getbatch()
                    if batch_v == False:
                        break
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)

                    val_loss = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    # batch_val_losses_main.append(val_loss_main)
                    # batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                # validation_loss_main = np.mean(batch_val_losses_main)
                # self.val_losses_main.append(validation_loss_main)
                # validation_loss_de = np.mean(batch_val_losses_de)
                # self.val_losses_de.append(validation_loss_de)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                , flush=True)
            # print(
            #         f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
            #     )

    def train_yearly(self, train_loader, val_loader, n_epochs=50, n_features=1, interv=50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []

            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...", flush=True)
            if timeCount:
                t_s = time.time()
                tt = time.time()
            for batch in train_loader:
                x_batch, y_batch, mc_mask, cc_mask = batch
                # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                # x_batch = x_batch[:, ~mask_ind, :]
                # y_batch = y_batch[:, ~mask_ind, :]
                x_batch_ = x_batch[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                mc_mask_ = mc_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                cc_mask_ = cc_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                y_batch_ = y_batch[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]

                years = int(x_batch_.size(1) / 365)
                x_batch_reshape = x_batch_.view([x_batch_.size(0) * years, 365, x_batch.size(2)])
                y_batch_reshape = y_batch_.view([y_batch_.size(0) * years, 365, y_batch.size(2)])
                mc_mask_reshape = mc_mask_.view([mc_mask_.size(0) * years, 365, mc_mask.size(2)])
                cc_mask_reshape = cc_mask_.view([cc_mask_.size(0) * years, 365, cc_mask.size(2)])
                # x_length[:] = 365
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch_reshape.shape[0]
                x_batch_reshape = x_batch_reshape.to(device)
                y_batch_reshape = y_batch_reshape.to(device)
                mc_mask_reshape = mc_mask_reshape.to(device)
                cc_mask_reshape = cc_mask_reshape.to(device)

                train_loss = 0
                slice = np.array(range(int(x_batch_reshape.size(0) / years))) * years
                for year in range(years):
                    x_tmp = x_batch_reshape[slice + year, :, :]
                    y_tmp = y_batch_reshape[slice + year, :, :]
                    mc_mask_tmp = mc_mask_reshape[slice + year, :, :]
                    cc_mask_tmp = cc_mask_reshape[slice + year, :, :]
                    if year == 0:
                        loss, hidden = self.train_step(x_tmp, y_tmp, mc_mask_tmp, cc_mask_tmp, hidden=None)
                    else:
                        loss, hidden = self.train_step(x_tmp, y_tmp, mc_mask_tmp, cc_mask_tmp, hidden=hidden)
                    train_loss += loss
                train_loss = train_loss / years
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('train taking %.3f s'%dif_t)
                batch_losses.append(train_loss)
                # batch_losses_main.append(loss_main)
                # batch_losses_de.append(loss_de)
                bN += 1

                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t = t_d - t_s
                        t_s = t_d
                        print('Taking %.3f s' % dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{len(train_loader)}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            # training_loss_main = np.mean(batch_losses_main)
            # self.train_losses_main.append(training_loss_main)
            # training_loss_de = np.mean(batch_losses_de)
            # self.train_losses_de.append(training_loss_de)

            # lr decay
            self.exp_lr_scheduler.step()

            # validation
            print(f"[{epoch}/{n_epochs}] Validating...", flush=True)

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                for batch_v in val_loader:
                    x_val, y_val, mc_mask, cc_mask = batch_v
                    # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                    # x_val = x_val[:, ~mask_ind, :]
                    # y_val = y_val[:, ~mask_ind, :]
                    mc_mask_ = mc_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    cc_mask_ = cc_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    x_val_ = x_val[~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1),
                             :, :]
                    y_val_ = y_val[~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1),
                             :, :]

                    years = int(x_val_.size(1) / 365)
                    x_val_reshape = x_val_.reshape(x_val_.size(0) * years, 365, x_val.size(2))
                    y_val_reshape = y_val_.reshape(y_val_.size(0) * years, 365, y_val.size(2))
                    mc_mask_reshape = mc_mask_.view([mc_mask_.size(0) * years, 365, mc_mask.size(2)])
                    cc_mask_reshape = cc_mask_.view([cc_mask_.size(0) * years, 365, cc_mask.size(2)])

                    # x_val_length[:] = 365
                    # print(torch.sum(torch.isnan(x_val_reshape)))
                    # print(torch.sum(torch.isnan(y_val_reshape)))
                    x_val_reshape = x_val_reshape.to(device)
                    y_val_reshape = y_val_reshape.to(device)
                    mc_mask_reshape = mc_mask_reshape.to(device)
                    cc_mask_reshape = cc_mask_reshape.to(device)
                    val_loss = 0
                    slice = np.array(range(int(x_val_reshape.size(0) / years))) * years
                    for year in range(years):
                        x_tmp = x_val_reshape[slice + year, :, :]
                        y_tmp = y_val_reshape[slice + year, :, :]
                        mc_mask_tmp = mc_mask_reshape[slice + year, :, :]
                        cc_mask_tmp = cc_mask_reshape[slice + year, :, :]
                        if year == 0:
                            loss, hidden_val = self.vali_step(x_tmp, y_tmp, mc_mask_tmp, cc_mask_tmp, hidden=None)
                        else:
                            loss, hidden_val = self.vali_step(x_tmp, y_tmp, mc_mask_tmp, cc_mask_tmp, hidden=hidden_val)
                        val_loss += loss
                    val_loss = val_loss / years
                    batch_val_losses.append(val_loss)
                    # batch_val_losses_main.append(val_loss_main)
                    # batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                # validation_loss_main = np.mean(batch_val_losses_main)
                # self.val_losses_main.append(validation_loss_main)
                # validation_loss_de = np.mean(batch_val_losses_de)
                # self.val_losses_de.append(validation_loss_de)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                , flush=True)
            # print(
            #         f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
            #     )

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            while True:
                batch = test_loader.getbatch()
                if batch == False:
                    break
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat, _ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def evaluate_yearly(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []

            for batch in test_loader:
                x_test, y_test, mc_mask, cc_mask = batch
                # print('eval', x_test.size(), y_test.size(), days_mask.size())
                # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                # x_test = x_test[:, ~mask_ind, :]
                # y_test = y_test[:, ~mask_ind, :]
                x_test_ = x_test[~torch.any(torch.any(torch.isnan(torch.cat((x_test, y_test), dim=2)), dim=2), dim=1),
                          :, :]
                y_test_ = y_test[~torch.any(torch.any(torch.isnan(torch.cat((x_test, y_test), dim=2)), dim=2), dim=1),
                          :, :]
                years = int(x_test_.size(1) / 365)
                x_test_reshape = x_test_.reshape(x_test_.shape[0] * years, 365, x_test_.shape[2])
                y_test_reshape = y_test_.reshape(y_test_.shape[0] * years, 365, y_test_.shape[2])
                # x_length[:] = 365
                x_test_reshape = x_test_reshape.to(device)
                y_test_reshape = y_test_reshape.to(device)

                self.model.eval()
                val_loss = 0
                yhat = torch.zeros(y_test_reshape.size()).to(device)
                slice = np.array(range(int(x_test_reshape.shape[0] / years))) * years
                for year in range(years):
                    x_tmp = x_test_reshape[slice + year, :, :]
                    # y_tmp = y_test_reshape[slice+year, :, :]
                    if year == 0:
                        yhat[slice + year, :, :], hidden, _ = self.model(x_tmp, hidden=None)
                    else:
                        yhat[slice + year, :, :], hidden, _ = self.model(x_tmp, hidden=hidden)

                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test_reshape.cpu().detach().numpy())

        return predictions, values

    def evaluate_singleStep(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            hidden_state = None
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch == False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat, hidden_state = self.model(x_test, hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())
        return predictions, values

    def plot_losses(self, outFolder=None, saveFig=False):
        fig = plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if saveFig:
            fig.savefig('%s/loss.png' % outFolder)


class Optimization_decoder_cropyear_365:
    def __init__(self, model, loss_fn=None, optimizer=None, exp_lr_scheduler=None, hiera=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.exp_lr_scheduler = exp_lr_scheduler
        self.train_losses = []
        self.val_losses = []
        self.train_losses_main = []
        self.val_losses_main = []
        self.train_losses_de = []
        self.val_losses_de = []
        self.hiera = hiera

    def changePadValue(self, y, seq_length):
        # replace the padding to -999
        t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                    batch_first=True, enforce_sorted=False)
        y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True, padding_value=-999)
        return y_repad.detach()

    def train_step(self, x, y, days_mask, mc_mask, cc_mask, hidden=None):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        if self.hiera:
            yhat, hidden, hTensor = self.model(x, hidden=hidden, seq_lengthList=days_mask, isTrain=True)
        else:
            yhat, hTensor = self.model(x, hidden=hidden, seq_lengthList=days_mask, isTrain=True)
        # print(x.size(), y.size(), yhat.size())
        # replace the padding to -999
        # y_repad = self.changePadValue(y=y,seq_length=x_length)

        loss_mc = torch.sum(((yhat - y) * mc_mask) ** 2) / torch.sum(mc_mask)
        loss_cc = torch.sum(((yhat - y) * cc_mask) ** 2) / torch.sum(cc_mask)
        loss = loss_mc + loss_cc
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item(), hidden

    def vali_step(self, x_val, y_val, days_mask, mc_mask, cc_mask, hidden=None):
        self.model.eval()
        yhat, hidden, hTensor = self.model(x_val, seq_lengthList=days_mask, isTrain=True)

        # mc_mask = days_mask.detach().clone()
        # mc_mask[:, 185:, :] = 0
        # cc_mask = days_mask.detach().clone()
        # cc_mask[:, :185, :] = 0
        loss_mc_val = torch.sum(((yhat - y_val) * mc_mask) ** 2) / torch.sum(mc_mask)
        loss_cc_val = torch.sum(((yhat - y_val) * cc_mask) ** 2) / torch.sum(cc_mask)
        val_loss = loss_mc_val + loss_cc_val

        return val_loss.item(), hidden

    def train(self, train_loader, val_loader, n_epochs=50, n_features=1, interv=50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []

            train_loader.epochStart()
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            while True:

                batch = train_loader.getbatch()
                if batch == False:
                    break
                x_batch, y_batch, x_length = batch
                if timeCount:
                    t_d = time.time()
                    dif_t = t_d - tt
                    tt = t_d
                    print('load batch taking %.3f s' % dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length)
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)
                if timeCount:
                    t_d = time.time()
                    dif_t = t_d - tt
                    tt = t_d
                    print('train taking %.3f s' % dif_t)
                bN += 1
                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t = t_d - t_s
                        t_s = t_d
                        print('Taking %.3f s' % dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{train_loader.epoch_batches}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)

            # lr decay
            self.exp_lr_scheduler.step()

            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                val_loader.epochStart()
                while True:
                    batch_v = val_loader.getbatch()
                    if batch_v == False:
                        break
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)

                    val_loss = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    # batch_val_losses_main.append(val_loss_main)
                    # batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                # validation_loss_main = np.mean(batch_val_losses_main)
                # self.val_losses_main.append(validation_loss_main)
                # validation_loss_de = np.mean(batch_val_losses_de)
                # self.val_losses_de.append(validation_loss_de)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                , flush=True)
            # print(
            #         f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
            #     )

    def train_yearly(self, train_loader, val_loader, n_epochs=50, n_features=1, interv=50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []

            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...", flush=True)
            if timeCount:
                t_s = time.time()
                tt = time.time()
            for batch in train_loader:
                x_batch, y_batch, days_mask, mc_mask, cc_mask = batch
                # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                # x_batch = x_batch[:, ~mask_ind, :]
                # y_batch = y_batch[:, ~mask_ind, :]
                x_batch_ = x_batch[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                days_mask_ = days_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                mc_mask_ = mc_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                cc_mask_ = cc_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                y_batch_ = y_batch[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]

                years = int(x_batch_.size(1) / 365)
                x_batch_reshape = x_batch_.view([x_batch_.size(0) * years, 365, x_batch.size(2)])
                y_batch_reshape = y_batch_.view([y_batch_.size(0) * years, 365, y_batch.size(2)])
                days_mask_reshape = days_mask_.reshape(days_mask_.shape[0] * years, 365, days_mask_.shape[2])
                mc_mask_reshape = mc_mask_.view([mc_mask_.size(0) * years, 365, mc_mask.size(2)])
                cc_mask_reshape = cc_mask_.view([cc_mask_.size(0) * years, 365, cc_mask.size(2)])
                # x_length[:] = 365
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch_reshape.shape[0]
                x_batch_reshape = x_batch_reshape.to(device)
                y_batch_reshape = y_batch_reshape.to(device)
                days_mask_reshape = days_mask_reshape.to(device)
                mc_mask_reshape = mc_mask_reshape.to(device)
                cc_mask_reshape = cc_mask_reshape.to(device)

                train_loss = 0
                slice = np.array(range(int(x_batch_reshape.size(0) / years))) * years
                for year in range(years):
                    x_tmp = x_batch_reshape[slice + year, :, :]
                    y_tmp = y_batch_reshape[slice + year, :, :]
                    days_mask_tmp = days_mask_reshape[slice + year, :, :]
                    mc_mask_tmp = mc_mask_reshape[slice + year, :, :]
                    cc_mask_tmp = cc_mask_reshape[slice + year, :, :]
                    if year == 0:
                        loss, hidden = self.train_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=None)
                    else:
                        loss, hidden = self.train_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=hidden)
                    train_loss += loss
                train_loss = train_loss / years
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('train taking %.3f s'%dif_t)
                batch_losses.append(train_loss)
                # batch_losses_main.append(loss_main)
                # batch_losses_de.append(loss_de)
                bN += 1

                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t = t_d - t_s
                        t_s = t_d
                        print('Taking %.3f s' % dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{len(train_loader)}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            # training_loss_main = np.mean(batch_losses_main)
            # self.train_losses_main.append(training_loss_main)
            # training_loss_de = np.mean(batch_losses_de)
            # self.train_losses_de.append(training_loss_de)

            # lr decay
            self.exp_lr_scheduler.step()

            # validation
            print(f"[{epoch}/{n_epochs}] Validating...", flush=True)

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                for batch_v in val_loader:
                    x_val, y_val, days_mask, mc_mask, cc_mask = batch_v
                    # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                    # x_val = x_val[:, ~mask_ind, :]
                    # y_val = y_val[:, ~mask_ind, :]
                    days_mask_ = days_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    mc_mask_ = mc_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    cc_mask_ = cc_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    x_val_ = x_val[~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1),
                             :, :]
                    y_val_ = y_val[~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1),
                             :, :]

                    years = int(x_val_.size(1) / 365)
                    x_val_reshape = x_val_.reshape(x_val_.size(0) * years, 365, x_val.size(2))
                    y_val_reshape = y_val_.reshape(y_val_.size(0) * years, 365, y_val.size(2))
                    days_mask_reshape = days_mask_.reshape(days_mask_.size(0) * years, 365, days_mask_.size(2))
                    mc_mask_reshape = mc_mask_.view([mc_mask_.size(0) * years, 365, mc_mask.size(2)])
                    cc_mask_reshape = cc_mask_.view([cc_mask_.size(0) * years, 365, cc_mask.size(2)])

                    # x_val_length[:] = 365
                    # print(torch.sum(torch.isnan(x_val_reshape)))
                    # print(torch.sum(torch.isnan(y_val_reshape)))
                    x_val_reshape = x_val_reshape.to(device)
                    y_val_reshape = y_val_reshape.to(device)
                    days_mask_reshape = days_mask_reshape.to(device)
                    mc_mask_reshape = mc_mask_reshape.to(device)
                    cc_mask_reshape = cc_mask_reshape.to(device)
                    val_loss = 0
                    slice = np.array(range(int(x_val_reshape.size(0) / years))) * years
                    for year in range(years):
                        x_tmp = x_val_reshape[slice + year, :, :]
                        y_tmp = y_val_reshape[slice + year, :, :]
                        days_mask_tmp = days_mask_reshape[slice + year, :, :]
                        mc_mask_tmp = mc_mask_reshape[slice + year, :, :]
                        cc_mask_tmp = cc_mask_reshape[slice + year, :, :]
                        if year == 0:
                            loss, hidden_val = self.vali_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=None)
                        else:
                            loss, hidden_val = self.vali_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=hidden_val)
                        val_loss += loss
                    val_loss = val_loss / years
                    batch_val_losses.append(val_loss)
                    # batch_val_losses_main.append(val_loss_main)
                    # batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                # validation_loss_main = np.mean(batch_val_losses_main)
                # self.val_losses_main.append(validation_loss_main)
                # validation_loss_de = np.mean(batch_val_losses_de)
                # self.val_losses_de.append(validation_loss_de)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                , flush=True)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            while True:
                batch = test_loader.getbatch()
                if batch == False:
                    break
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat, _ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def evaluate_yearly(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []

            for batch in test_loader:
                x_test, y_test, days_mask, mc_mask, cc_mask = batch
                # print('eval', x_test.size(), y_test.size(), days_mask.size())
                # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                # x_test = x_test[:, ~mask_ind, :]
                # y_test = y_test[:, ~mask_ind, :]
                x_test_ = x_test[~torch.any(torch.any(torch.isnan(torch.cat((x_test, y_test), dim=2)), dim=2), dim=1),
                          :, :]
                y_test_ = y_test[~torch.any(torch.any(torch.isnan(torch.cat((x_test, y_test), dim=2)), dim=2), dim=1),
                          :, :]
                years = int(x_test_.size(1) / 365)
                x_test_reshape = x_test_.reshape(x_test_.shape[0] * years, 365, x_test_.shape[2])
                y_test_reshape = y_test_.reshape(y_test_.shape[0] * years, 365, y_test_.shape[2])
                # x_length[:] = 365
                x_test_reshape = x_test_reshape.to(device)
                y_test_reshape = y_test_reshape.to(device)

                self.model.eval()
                val_loss = 0
                yhat = torch.zeros(y_test_reshape.size()).to(device)
                slice = np.array(range(int(x_test_reshape.shape[0] / years))) * years
                for year in range(years):
                    x_tmp = x_test_reshape[slice + year, :, :]
                    # y_tmp = y_test_reshape[slice+year, :, :]
                    if year == 0:
                        yhat[slice + year, :, :], hidden, _ = self.model(x_tmp, hidden=None)
                    else:
                        yhat[slice + year, :, :], hidden, _ = self.model(x_tmp, hidden=hidden)

                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test_reshape.cpu().detach().numpy())

        return predictions, values

    def evaluate_singleStep(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            hidden_state = None
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch == False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat, hidden_state = self.model(x_test, hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())
        return predictions, values

    def plot_losses(self, outFolder=None, saveFig=False):
        fig = plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if saveFig:
            fig.savefig('%s/loss.png' % outFolder)


class Optimization_decoder_cropyear:
    def __init__(self, model, loss_fn=None, optimizer=None, exp_lr_scheduler=None, hiera=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.exp_lr_scheduler = exp_lr_scheduler
        self.train_losses = []
        self.val_losses = []
        self.train_losses_main = []
        self.val_losses_main = []
        self.train_losses_de = []
        self.val_losses_de = []
        self.hiera = hiera

    def changePadValue(self, y, seq_length):
        # replace the padding to -999
        t = torch.nn.utils.rnn.pack_padded_sequence(y, seq_length,
                                                    batch_first=True, enforce_sorted=False)
        y_repad, _ = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True, padding_value=-999)
        return y_repad.detach()

    def train_step(self, x, y, days_mask, mc_mask, cc_mask, hidden=None):
        # Sets model to train mode
        self.model.train()
        # Makes predictions
        if self.hiera:
            yhat, hidden, hTensor = self.model(x, hidden=hidden, seq_lengthList=days_mask, isTrain=True)
        else:
            yhat, hTensor = self.model(x, hidden=hidden, seq_lengthList=days_mask, isTrain=True)
        yhat, _ = pad_packed_sequence(yhat, batch_first=True)
        # print(x.size(), y.size(), yhat.size())
        # replace the padding to -999
        # y_repad = self.changePadValue(y=y,seq_length=x_length)

        loss_mc = torch.sum(((yhat - y) * mc_mask) ** 2) / torch.sum(mc_mask)
        loss_cc = torch.sum(((yhat - y) * cc_mask) ** 2) / torch.sum(cc_mask)
        loss = loss_mc + loss_cc
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item(), hidden

    def vali_step(self, x_val, y_val, days_mask, mc_mask, cc_mask, hidden=None):
        self.model.eval()
        yhat, hidden, hTensor = self.model(x_val, seq_lengthList=days_mask, isTrain=True)
        yhat, _ = pad_packed_sequence(yhat, batch_first=True)
        # mc_mask = days_mask.detach().clone()
        # mc_mask[:, 185:, :] = 0
        # cc_mask = days_mask.detach().clone()
        # cc_mask[:, :185, :] = 0
        loss_mc_val = torch.sum(((yhat - y_val) * mc_mask) ** 2) / torch.sum(mc_mask)
        loss_cc_val = torch.sum(((yhat - y_val) * cc_mask) ** 2) / torch.sum(cc_mask)
        val_loss = loss_mc_val + loss_cc_val

        return val_loss.item(), hidden

    def train(self, train_loader, val_loader, n_epochs=50, n_features=1, interv=50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []

            train_loader.epochStart()
            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...")
            if timeCount:
                t_s = time.time()
                tt = time.time()
            while True:

                batch = train_loader.getbatch()
                if batch == False:
                    break
                x_batch, y_batch, x_length = batch
                if timeCount:
                    t_d = time.time()
                    dif_t = t_d - tt
                    tt = t_d
                    print('load batch taking %.3f s' % dif_t)
                batch_size = x_batch.shape[0]
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss, loss_main, loss_de = self.train_step(x_batch, y_batch, x_length)
                batch_losses.append(loss)
                batch_losses_main.append(loss_main)
                batch_losses_de.append(loss_de)
                if timeCount:
                    t_d = time.time()
                    dif_t = t_d - tt
                    tt = t_d
                    print('train taking %.3f s' % dif_t)
                bN += 1
                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t = t_d - t_s
                        t_s = t_d
                        print('Taking %.3f s' % dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{train_loader.epoch_batches}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            training_loss_main = np.mean(batch_losses_main)
            self.train_losses_main.append(training_loss_main)
            training_loss_de = np.mean(batch_losses_de)
            self.train_losses_de.append(training_loss_de)

            # lr decay
            self.exp_lr_scheduler.step()

            # validation
            print(f"[{epoch}/{n_epochs}] Validating...")

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                val_loader.epochStart()
                while True:
                    batch_v = val_loader.getbatch()
                    if batch_v == False:
                        break
                    x_val, y_val, x_val_length = batch_v
                    batch_size_v = x_val.shape[0]
                    x_val = x_val.view([batch_size_v, -1, n_features]).to(device)
                    y_val = y_val.to(device)

                    val_loss = self.vali_step(x_val, y_val, x_val_length)
                    batch_val_losses.append(val_loss)
                    # batch_val_losses_main.append(val_loss_main)
                    # batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                # validation_loss_main = np.mean(batch_val_losses_main)
                # self.val_losses_main.append(validation_loss_main)
                # validation_loss_de = np.mean(batch_val_losses_de)
                # self.val_losses_de.append(validation_loss_de)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                , flush=True)
            # print(
            #         f"[{epoch}/{n_epochs}] Training loss_main/decoder: [{training_loss_main:.4f}/{training_loss_de:.4f}]\t Validation loss_main/decoder: [{validation_loss_main:.4f}/{validation_loss_de:.4f}] "
            #     )

    def train_yearly(self, train_loader, val_loader, n_epochs=50, n_features=1, interv=50, timeCount=False):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            batch_losses_main = []
            batch_losses_de = []

            bN = 0
            print(f"[{epoch}/{n_epochs}] Training...", flush=True)
            if timeCount:
                t_s = time.time()
                tt = time.time()
            for batch in train_loader:
                x_batch, y_batch, days_mask, mc_mask, cc_mask = batch
                # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                # x_batch = x_batch[:, ~mask_ind, :]
                # y_batch = y_batch[:, ~mask_ind, :]
                x_batch_ = x_batch[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                days_mask_ = days_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                mc_mask_ = mc_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                cc_mask_ = cc_mask[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]
                y_batch_ = y_batch[
                           ~torch.any(torch.any(torch.isnan(torch.cat((x_batch, y_batch), dim=2)), dim=2), dim=1), :, :]

                years = int(x_batch_.size(1) / 365)
                x_batch_reshape = x_batch_.view([x_batch_.size(0) * years, 365, x_batch.size(2)])
                y_batch_reshape = y_batch_.view([y_batch_.size(0) * years, 365, y_batch.size(2)])
                days_mask_reshape = days_mask_.reshape(days_mask_.shape[0] * years, 365, days_mask_.shape[2])
                mc_mask_reshape = mc_mask_.view([mc_mask_.size(0) * years, 365, mc_mask.size(2)])
                cc_mask_reshape = cc_mask_.view([cc_mask_.size(0) * years, 365, cc_mask.size(2)])
                # x_length[:] = 365
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('load batch taking %.3f s'%dif_t)
                batch_size = x_batch_reshape.shape[0]
                x_batch_reshape = x_batch_reshape.to(device)
                y_batch_reshape = y_batch_reshape.to(device)
                days_mask_reshape = days_mask_reshape.to(device)
                mc_mask_reshape = mc_mask_reshape.to(device)
                cc_mask_reshape = cc_mask_reshape.to(device)

                train_loss = 0
                slice = np.array(range(int(x_batch_reshape.size(0) / years))) * years
                for year in range(years):
                    x_tmp = x_batch_reshape[slice + year, :, :]
                    y_tmp = y_batch_reshape[slice + year, :, :]
                    days_mask_tmp = days_mask_reshape[slice + year, :, :]
                    mc_mask_tmp = mc_mask_reshape[slice + year, :, :]
                    cc_mask_tmp = cc_mask_reshape[slice + year, :, :]
                    if year == 0:
                        loss, hidden = self.train_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=None)
                    else:
                        loss, hidden = self.train_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=hidden)
                    train_loss += loss
                train_loss = train_loss / years
                # if timeCount:
                #     t_d = time.time()
                #     dif_t =  t_d-tt
                #     tt = t_d
                #     print('train taking %.3f s'%dif_t)
                batch_losses.append(train_loss)
                # batch_losses_main.append(loss_main)
                # batch_losses_de.append(loss_de)
                bN += 1

                if bN % interv == 0:
                    if timeCount:
                        t_d = time.time()
                        dif_t = t_d - t_s
                        t_s = t_d
                        print('Taking %.3f s' % dif_t)
                    print(f"[{epoch}/{n_epochs}] finished [{bN}/{len(train_loader)}] batches, ")
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            # training_loss_main = np.mean(batch_losses_main)
            # self.train_losses_main.append(training_loss_main)
            # training_loss_de = np.mean(batch_losses_de)
            # self.train_losses_de.append(training_loss_de)

            # lr decay
            self.exp_lr_scheduler.step()

            # validation
            print(f"[{epoch}/{n_epochs}] Validating...", flush=True)

            with torch.no_grad():
                batch_val_losses = []
                batch_val_losses_main = []
                batch_val_losses_de = []
                for batch_v in val_loader:
                    x_val, y_val, days_mask, mc_mask, cc_mask = batch_v
                    # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                    # x_val = x_val[:, ~mask_ind, :]
                    # y_val = y_val[:, ~mask_ind, :]
                    days_mask_ = days_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    mc_mask_ = mc_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    cc_mask_ = cc_mask[
                               ~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1), :, :]
                    x_val_ = x_val[~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1),
                             :, :]
                    y_val_ = y_val[~torch.any(torch.any(torch.isnan(torch.cat((x_val, y_val), dim=2)), dim=2), dim=1),
                             :, :]

                    years = int(x_val_.size(1) / 365)
                    x_val_reshape = x_val_.reshape(x_val_.size(0) * years, 365, x_val.size(2))
                    y_val_reshape = y_val_.reshape(y_val_.size(0) * years, 365, y_val.size(2))
                    days_mask_reshape = days_mask_.reshape(days_mask_.size(0) * years, 365, days_mask_.size(2))
                    mc_mask_reshape = mc_mask_.view([mc_mask_.size(0) * years, 365, mc_mask.size(2)])
                    cc_mask_reshape = cc_mask_.view([cc_mask_.size(0) * years, 365, cc_mask.size(2)])

                    # x_val_length[:] = 365
                    # print(torch.sum(torch.isnan(x_val_reshape)))
                    # print(torch.sum(torch.isnan(y_val_reshape)))
                    x_val_reshape = x_val_reshape.to(device)
                    y_val_reshape = y_val_reshape.to(device)
                    days_mask_reshape = days_mask_reshape.to(device)
                    mc_mask_reshape = mc_mask_reshape.to(device)
                    cc_mask_reshape = cc_mask_reshape.to(device)
                    val_loss = 0
                    slice = np.array(range(int(x_val_reshape.size(0) / years))) * years
                    for year in range(years):
                        x_tmp = x_val_reshape[slice + year, :, :]
                        y_tmp = y_val_reshape[slice + year, :, :]
                        days_mask_tmp = days_mask_reshape[slice + year, :, :]
                        mc_mask_tmp = mc_mask_reshape[slice + year, :, :]
                        cc_mask_tmp = cc_mask_reshape[slice + year, :, :]
                        if year == 0:
                            loss, hidden_val = self.vali_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=None)
                        else:
                            loss, hidden_val = self.vali_step(x_tmp, y_tmp, days_mask_tmp, mc_mask_tmp, cc_mask_tmp, hidden=hidden_val)
                        val_loss += loss
                    val_loss = val_loss / years
                    batch_val_losses.append(val_loss)
                    # batch_val_losses_main.append(val_loss_main)
                    # batch_val_losses_de.append(val_loss_de)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                # validation_loss_main = np.mean(batch_val_losses_main)
                # self.val_losses_main.append(validation_loss_main)
                # validation_loss_de = np.mean(batch_val_losses_de)
                # self.val_losses_de.append(validation_loss_de)

            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f} lr: {self.exp_lr_scheduler.get_last_lr()[0]:.6f}"
                , flush=True)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            while True:
                batch = test_loader.getbatch()
                if batch == False:
                    break
                x_test, y_test, x_length = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat, _ = self.model(x_test)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())

        return predictions, values

    def evaluate_yearly(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []

            for batch in test_loader:
                x_test, y_test, days_mask, mc_mask, cc_mask = batch
                # print('eval', x_test.size(), y_test.size(), days_mask.size())
                # mask_ind = (pd.date_range('2000-01-01', '2020-12-31', freq='D').dayofyear == 366)
                # x_test = x_test[:, ~mask_ind, :]
                # y_test = y_test[:, ~mask_ind, :]
                x_test_ = x_test[~torch.any(torch.any(torch.isnan(torch.cat((x_test, y_test), dim=2)), dim=2), dim=1),
                          :, :]
                y_test_ = y_test[~torch.any(torch.any(torch.isnan(torch.cat((x_test, y_test), dim=2)), dim=2), dim=1),
                          :, :]
                years = int(x_test_.size(1) / 365)
                x_test_reshape = x_test_.reshape(x_test_.shape[0] * years, 365, x_test_.shape[2])
                y_test_reshape = y_test_.reshape(y_test_.shape[0] * years, 365, y_test_.shape[2])
                # x_length[:] = 365
                x_test_reshape = x_test_reshape.to(device)
                y_test_reshape = y_test_reshape.to(device)

                self.model.eval()
                val_loss = 0
                yhat = torch.zeros(y_test_reshape.size()).to(device)
                slice = np.array(range(int(x_test_reshape.shape[0] / years))) * years
                for year in range(years):
                    x_tmp = x_test_reshape[slice + year, :, :]
                    # y_tmp = y_test_reshape[slice+year, :, :]
                    if year == 0:
                        yhat[slice + year, :, :], hidden, _ = self.model(x_tmp, hidden=None)
                    else:
                        yhat[slice + year, :, :], hidden, _ = self.model(x_tmp, hidden=hidden)

                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test_reshape.cpu().detach().numpy())

        return predictions, values

    def evaluate_singleStep(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            test_loader.epochStart()
            hidden_state = None
            while True:
                if test_loader.newEpisode == True:
                    hidden_state = None
                batch = test_loader.getbatch()
                if batch == False:
                    break
                x_test, y_test = batch
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat, hidden_state = self.model(x_test, hidden_state)
                predictions.append(yhat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())
        return predictions, values

    def plot_losses(self, outFolder=None, saveFig=False):
        fig = plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if saveFig:
            fig.savefig('%s/loss.png' % outFolder)


class EcoSOC_dataset_pkl(Dataset):
    def __init__(self, data_pathes, X_Features=None, y_Features=None, scaler_inputs=None, scaler_outputs=None):
        self.data_pathes = data_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_Features = X_Features
        self.y_Features = y_Features
        self.scaler_inputs = scaler_inputs
        self.scaler_outputs = scaler_outputs

    def __getitem__(self, index):

        self.data = self.load_object(self.data_pathes[index])
        self.data = self.data[self.data['DOY'] != 366].reset_index(drop=True)
        self.length = self.data.shape[0]
        # self.data = self.data.dropna().reset_index(drop=True)

        # if self.data.shape[0] == self.length:
        out_X = np.array(self.data[self.X_Features]).astype(np.float32)
        out_y = np.array(self.data[self.y_Features]).astype(np.float32)

        if not self.scaler_inputs.any() == None:
            for i, scaler in enumerate(self.scaler_inputs):
                X_mean, X_std = scaler[0], scaler[1]
                out_X[:, i] = (out_X[:, i] - X_mean) / X_std

        if not self.scaler_outputs.any() == None:
            for i, scaler in enumerate(self.scaler_outputs):
                y_mean, y_std = scaler[0], scaler[1]
                out_y[:, i] = (out_y[:, i] - y_mean) / y_std

        years = int(self.length / 365)
        ann_y = np.zeros([years, len(self.y_Features)], dtype=np.float32)
        for i in range(years):
            if len(self.y_Features) == 2:
                ann_y[i, 0] = np.max(out_y[365 * i:365 * (i + 1), 0]) # yield
                ann_y[i, 1] = out_y[365 * i + 364, 1] - out_y[365 * i, 1] # SOC
            else:
                ann_y[i, 0] = out_y[365 * i + 364, 0] # surface residue
                ann_y[i, 1] = out_y[365 * i + 364, 1] # soil residue
                ann_y[i, 2] = np.max(out_y[365 * i:365 * (i + 1), 2])  # yield
                ann_y[i, 3] = out_y[365 * i + 364, 3] - out_y[365 * i, 3]  # SOC


        return out_X, out_y, ann_y
        # else:
        #     return np.array([], dtype=np.float32).reshape(0, len(self.X_Features)),\
        #            np.array([], dtype=np.float32).reshape(0, len(self.y_Features)),\
        #            np.array([], dtype=np.float32).reshape(0, 2)


    def load_object(self, filename):
        # data = pd.read_csv(filename)
        # return data
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_pathes)


class EcoSOC_dataset_pkl_maxres(Dataset):
    def __init__(self, data_pathes, X_Features=None, y_Features=None, scaler_inputs=None, scaler_outputs=None):
        self.data_pathes = data_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_Features = X_Features
        self.y_Features = y_Features
        self.scaler_inputs = scaler_inputs
        self.scaler_outputs = scaler_outputs

    def __getitem__(self, index):

        self.data = self.load_object(self.data_pathes[index])
        self.data = self.data[self.data['DOY'] != 366].reset_index(drop=True)
        self.length = self.data.shape[0]
        # self.data = self.data.dropna().reset_index(drop=True)

        # if self.data.shape[0] == self.length:
        out_X = np.array(self.data[self.X_Features]).astype(np.float32)
        out_y = np.array(self.data[self.y_Features]).astype(np.float32)

        if not self.scaler_inputs.any() == None:
            for i, scaler in enumerate(self.scaler_inputs):
                X_mean, X_std = scaler[0], scaler[1]
                out_X[:, i] = (out_X[:, i] - X_mean) / X_std

        if not self.scaler_outputs.any() == None:
            for i, scaler in enumerate(self.scaler_outputs):
                y_mean, y_std = scaler[0], scaler[1]
                out_y[:, i] = (out_y[:, i] - y_mean) / y_std

        years = int(self.length / 365)
        ann_y = np.zeros([years, len(self.y_Features)], dtype=np.float32)
        for i in range(years):
            if len(self.y_Features) == 2:
                ann_y[i, 0] = np.max(out_y[365 * i:365 * (i + 1), 0]) # yield
                ann_y[i, 1] = out_y[365 * i + 364, 1] - out_y[365 * i, 1] # SOC
            else:
                ann_y[i, 0] = np.max(out_y[365 * i:365 * (i + 1), 0]) # surface residue
                ann_y[i, 1] = np.max(out_y[365 * i:365 * (i + 1), 1]) # soil residue
                ann_y[i, 2] = np.max(out_y[365 * i:365 * (i + 1), 2])  # yield
                ann_y[i, 3] = out_y[365 * i + 364, 3] - out_y[365 * i, 3]  # SOC


        return out_X, out_y, ann_y
        # else:
        #     return np.array([], dtype=np.float32).reshape(0, len(self.X_Features)),\
        #            np.array([], dtype=np.float32).reshape(0, len(self.y_Features)),\
        #            np.array([], dtype=np.float32).reshape(0, 2)


    def load_object(self, filename):
        # data = pd.read_csv(filename)
        # return data
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_pathes)


class EcoSOC_dataset_pkl_cropyear(Dataset):
    def __init__(self, data_pathes, X_Features=None, y_Features=None, scaler_inputs=None, scaler_outputs=None):
        self.data_pathes = data_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_Features = X_Features
        self.y_Features = y_Features
        self.scaler_inputs = scaler_inputs
        self.scaler_outputs = scaler_outputs

    def __getitem__(self, index):

        self.data = self.load_object(self.data_pathes[index])
        self.data = self.data[self.data['DOY'] != 366].reset_index(drop=True)
        self.length = self.data.shape[0]
        # self.data = self.data.dropna().reset_index(drop=True)
        max_gslength = 390 # 365 d add with 25 d planting disturbance
        years = 20
        out_X = np.zeros(([max_gslength * years, len(self.X_Features)])).astype(np.float32)
        out_y = np.zeros(([max_gslength * years, len(self.y_Features)])).astype(np.float32)
        seq_length = np.zeros(([years])).astype(np.int)
        ann_y = np.zeros([years, len(self.y_Features)], dtype=np.float32)
        for i, year in enumerate(range(2000, 2020)):
            gs_start = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year))].index
            gs_end = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year + 1))].index

            if len(gs_start) > 0 and len(gs_end) > 0:
                crop_year = self.data.loc[gs_start[0]:gs_end[0]-1, :]
            elif len(gs_start) > 0:
                crop_year = self.data.loc[gs_start[0]:(gs_start[0] + max_gslength-1), :]
            else:
                continue
            days = crop_year.shape[0]
            seq_length[i] = days
            out_X[max_gslength * i:(max_gslength * i + days), :] = crop_year[self.X_selectFeatures].to_numpy()
            out_y[max_gslength * i:(max_gslength * i + days), :] = crop_year[self.y_selectFeatures].to_numpy()
            # extract annual variables
            if len(self.y_Features) == 2:
                ann_y[i, 0] = np.max(out_y[max_gslength * i:(max_gslength * i + days), 0]) # yield
                ann_y[i, 1] = out_y[max_gslength * i + days, 1] - out_y[max_gslength * i, 1] # SOC
            else:
                ann_y[i, 0] = out_y[max_gslength * i + days, 0] # surface residue
                ann_y[i, 1] = out_y[max_gslength * i + days, 1] # soil residue
                ann_y[i, 2] = np.max(out_y[max_gslength * i:(max_gslength * i + days), 2])  # yield
                ann_y[i, 3] = out_y[max_gslength * i + days, 3] - out_y[max_gslength * i, 3]  # SOC

        if not self.scaler_inputs.any() == None:
            for i, scaler in enumerate(self.scaler_inputs):
                X_mean, X_std = scaler[0], scaler[1]
                out_X[:, i] = (out_X[:, i] - X_mean) / X_std

        if not self.scaler_outputs.any() == None:
            for i, scaler in enumerate(self.scaler_outputs):
                y_mean, y_std = scaler[0], scaler[1]
                out_y[:, i] = (out_y[:, i] - y_mean) / y_std
                ann_y[:, i] = (ann_y[:, i] - y_mean) / y_std


        return out_X, out_y, ann_y, seq_length


    def load_object(self, filename):
        # data = pd.read_csv(filename)
        # return data
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_pathes)


class EcoSOC_dataset_pkl_maxres_cropyear(Dataset):
    def __init__(self, data_pathes, X_Features=None, y_Features=None, scaler_inputs=None, scaler_outputs=None):
        self.data_pathes = data_pathes
        # self.data_y_pathes = data_y_pathes
        self.X_Features = X_Features
        self.y_Features = y_Features
        self.scaler_inputs = scaler_inputs
        self.scaler_outputs = scaler_outputs

    def __getitem__(self, index):

        self.data = self.load_object(self.data_pathes[index])
        self.data = self.data[self.data['DOY'] != 366].reset_index(drop=True)
        self.length = self.data.shape[0]
        # self.data = self.data.dropna().reset_index(drop=True)

        max_gslength = 390 # 365 d add with 25 d planting disturbance
        years = 20
        out_X = np.zeros(([max_gslength * years, len(self.X_Features)])).astype(np.float32)
        out_y = np.zeros(([max_gslength * years, len(self.y_Features)])).astype(np.float32)
        seq_length = np.zeros(([years])).astype(np.int)
        ann_y = np.zeros([years, len(self.y_Features)], dtype=np.float32)
        for i, year in enumerate(range(2000, 2020)):
            gs_start = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year))].index
            gs_end = self.data[(self.data['Growseason'] > 0) & (
                        self.data['DATE'].astype(int).astype(str).str[-4:] == str(year + 1))].index

            if len(gs_start) > 0 and len(gs_end) > 0:
                crop_year = self.data.loc[gs_start[0]:gs_end[0]-1, :]
            elif len(gs_start) > 0:
                crop_year = self.data.loc[gs_start[0]:(gs_start[0] + 389), :]
            else:
                continue
            days = crop_year.shape[0]
            seq_length[i] = days
            out_X[max_gslength * i:(max_gslength * i + days), :] = crop_year[self.X_selectFeatures].to_numpy()
            out_y[max_gslength * i:(max_gslength * i + days), :] = crop_year[self.y_selectFeatures].to_numpy()
            # extract annual variables
            if len(self.y_Features) == 2:
                ann_y[i, 0] = np.max(out_y[max_gslength * i:(max_gslength * i + days), 0]) # yield
                ann_y[i, 1] = out_y[max_gslength * i + days, 1] - out_y[max_gslength * i, 1] # SOC
            else:
                ann_y[i, 0] = np.max(out_y[max_gslength * i:(max_gslength * i + days), 0]) # surface residue
                ann_y[i, 1] = np.max(out_y[max_gslength * i:(max_gslength * i + days), 1]) # soil residue
                ann_y[i, 2] = np.max(out_y[max_gslength * i:(max_gslength * i + days), 2])  # yield
                ann_y[i, 3] = out_y[max_gslength * i + days, 3] - out_y[max_gslength * i, 3]  # SOC

        if not self.scaler_inputs.any() == None:
            for i, scaler in enumerate(self.scaler_inputs):
                X_mean, X_std = scaler[0], scaler[1]
                out_X[:, i] = (out_X[:, i] - X_mean) / X_std

        if not self.scaler_outputs.any() == None:
            for i, scaler in enumerate(self.scaler_outputs):
                y_mean, y_std = scaler[0], scaler[1]
                out_y[:, i] = (out_y[:, i] - y_mean) / y_std
                ann_y[:, i] = (ann_y[:, i] - y_mean) / y_std


        return out_X, out_y, ann_y, seq_length


    def load_object(self, filename):
        # data = pd.read_csv(filename)
        # return data
        with open(filename, 'rb') as inp:
            data = pickle.load(inp)
        return data

    def __len__(self):
        return len(self.data_pathes)


def plot_losses(losses, loss_names, outFolder=None):
    train_losses = losses['train_losses']
    val_losses = losses['val_losses']
    n = train_losses.shape[0]
    fig, ax = plt.subplots(n,2,figsize=(7*2, 5*4))
    train_plots = np.zeros([n,train_losses.shape[1]])
    val_plots = np.zeros([n,val_losses.shape[1]])
    for i in range(n):
        train_plots[i,:] = train_losses[i,:]
        val_plots[i,:] = val_losses[i,:]


    for i in range(2):
        for j in range(n):
            zoomin = round(max(train_plots[j,:][-1], val_plots[j,:][-1]) + 0.1, 1)
            ax[j,i].plot(train_plots[j,:], label="Train loss"+str(j))
            ax[j,i].plot(val_plots[j,:], label="Val loss"+str(j))
            ax[j,i].set_ylabel(loss_names[j])
            if i==1:
                ax[j,i].set_ylim([0,zoomin])
                ax[j,i].set_xlabel("Epoch")
            ax[j,i].legend()
    fig.savefig('%s/model_losses.png' % (outFolder), dpi=500)



def plotScatterDense(x_, y_, alpha=1, binN=200, thresh_p=None, outFolder='',
                     saveFig=False, note='', title='', uplim=None, downlim=None, auxText=None, legendLoc=4):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.rcParams.update({'font.size': 14})
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

    plt.scatter(x2, y2, c=z2, cmap='Blues', marker='.', alpha=alpha, vmin=0, vmax=50)

    if uplim == None:
        uplim = 1.2 * max(np.hstack((x, y)))
    if downlim == None:
        downlim = 0.8 * min(np.hstack((x, y)))

    figRange = uplim - downlim
    plt.plot(np.arange(np.ceil(downlim) - 1, np.ceil(uplim) + 1), np.arange(np.ceil(downlim) - 1, np.ceil(uplim) + 1),
             'k', label='1:1 line')
    # plt.xlim([downlim, uplim])
    # plt.ylim([downlim, uplim])
    plt.xlabel('Synthetic data', fontsize=16)
    plt.ylabel('Predictions', fontsize=16)
    plt.axis('square')
    # if note == 'NEE':
    #     plt.xlim([-20, uplim])
    #     plt.ylim([-20, uplim])
    # else:
    #     plt.xlim([0, uplim])
    #     plt.ylim([0, uplim])
    if not legendLoc is None:
        if legendLoc == False:
            plt.legend(edgecolor='w', facecolor='w', fontsize=13)
        else:
            plt.legend(loc=legendLoc, edgecolor='w', facecolor='w', fontsize=13, framealpha=0)
    # plt.title(title, y=0.9)

    if len(y) > 1:
        R2 = np.corrcoef(x, y)[0, 1] ** 2
        RMSE = (np.sum((y - x) ** 2) / len(y)) ** 0.5
        # MAPE = 100 * np.sum(np.abs((y - x) / (x+0.00001))) / len(x)
        plt.text(downlim + 0.1 * figRange, downlim + 0.83 * figRange, r'$R^2$ = ' + str(R2)[:5], fontsize=15)
        # ax = plt.text(0.1 * uplim, 0.86 * uplim, r'$MAPE $= ' + str(MAPE)[:5] + '%', fontsize=14)
        plt.text(downlim + 0.1 * figRange, downlim + 0.76 * figRange, r'$RMSE$ = ' + str(RMSE)[:5], fontsize=15)

        plt.text(downlim + 0.1 * figRange, downlim + 0.69 * figRange, r'$Slope$ = ' + str(para[0])[:5], fontsize=15)
    if not auxText == None:
        plt.text(0.05, 0.9, auxText, transform=ax.transAxes, fontproperties='Times New Roman', fontsize=15)
    plt.colorbar()

    if saveFig:
        plt.title(title, y=0.9, fontsize=16)
        fig.savefig('%s/test_scatter_%s.png' % (outFolder, note), dpi=350)

    else:
        plt.title(title)