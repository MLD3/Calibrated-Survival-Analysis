import numpy as np
import pandas as pd
import torch as torch
import pickle as pickle
import bisect
from collections import Counter
import argparse
import os
import shutil
import scipy as sp
from torch.utils.data import Dataset

class SurvivalDataset(Dataset):
    '''Create dataset of covariates, censorship status, and survival'''
    def __init__(self, covs, censored, survival):
        self.covs = torch.tensor(covs)
        self.censored = torch.tensor(censored)
        self.survival = torch.tensor(survival)
        
    def __len__(self):
        return len(self.covs)
        
    def __getitem__(self, idx):
        return self.covs[idx], self.censored[idx], self.survival[idx]


def binned_dist(survival_curves, times, censored, bins = 10):
    '''Bin estimated survival probabilities at true time points into "bins"
     equal length intervals between 0 and 1 '''
    bins = np.append(np.arange(0, 1.00, 1/bins), [1])
    survival_probs = [survival_curves[i][times[i]] for i in range(len(censored)) if not censored[i]]
    intervals = np.array([bisect.bisect_right(bins, p) for p in survival_probs])
    counts = Counter(intervals)
    if bins+1 in counts:
        counts[bins] += counts[bins+1]
        del counts[bins+1]
    #Lump together 0 survival probability bin with first bin
    if 0 in counts:
        counts[1] += counts[0]
        del counts[0]
    for num in range(1,bins+1,1):
        if num not in counts:
            counts[num] = 0
    
    return [counts[val] for val in sorted(counts)]

def ddc(p, q):
    '''Calculate DDC between two binned arrays'''
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    ### Calculate entropy between two. DDC can be measured with any divergence measure, 
    #such as KL or others
    return sp.stats.entropy(p,q, base=10)

def return_split(dataset, batch_size, seed):
        '''Return datasets
        Change this to however you want to split your data 
        In the end, it should return: 
        train_loader, validation_loader, test_loader, train_mean, train_std, t, cov_size
            -- data.DataLoader for your train, validation, and test sets 
            --- mean and std of train covariates to normalize data
            --- Time points to supervise over (max time) 
            --- Size of covariates in data
        
        Following code is for example data with CLINIC and NACD, can be changed 
        to any dataset of interest for which data is saved nearby 

         '''
    if dataset == 'CLINIC':
        train = pd.read_csv('data/CLINIC/train.log.txt', sep = '\t')
        test = pd.read_csv('data/CLINIC/test.log.txt', sep = '\t')

        survival = train.append(test)
        survival['event'] = 1 - survival['event']
        survival['time'] = ((survival['time']).astype(int))
        train, test = train_test_split(survival, test_size = .4, random_state = seed, shuffle = True, stratify = survival[['event']])
        valid, test = train_test_split(test, test_size = .5, random_state = seed, shuffle = True, stratify = test[['event']])

        #Comment in if validating on only uncensored individuals
        # valid = val.loc[val['event'] == 0]

        train_survival_times = train.values[:,-2]
        train_censored = train.values[:,-1]
        train_covs = train.values[:,0:-2]

        valid_survival_times = valid.values[:,-2]
        valid_censored = valid.values[:,-1]
        valid_covs = valid.values[:,0:-2]

        test_survival_times = test.values[:,-2]
        test_censored = test.values[:,-1]
        test_covs = test.values[:,0:-2]

        train_loader = data.DataLoader(SurvivalDataset(train_covs, train_censored, train_survival_times), batch_size = batch_size, shuffle = True)
        validation_loader = data.DataLoader(SurvivalDataset(valid_covs, valid_censored, valid_survival_times), batch_size = len(valid_covs))
        test_loader = data.DataLoader(SurvivalDataset(test_covs, test_censored, test_survival_times), batch_size = len(test_covs))

        train_mean = torch.tensor(train_covs.mean(axis=0)).float()
        train_std = torch.tensor(train_covs.std(axis = 0)).float()

        t = 52
        cov_size = 15
    else:
        survival = pd.read_csv('data/NACD/data.csv')
        survival['SURVIVAL'] = ((survival['SURVIVAL']-1e-5).astype(int))+1
        train, test = train_test_split(survival, test_size = .4, random_state = seed, shuffle = True, stratify = survival[['CENSORED']])
        valid, test = train_test_split(test, test_size = .5, random_state = seed, shuffle = True, stratify = test[['CENSORED']])

        #Comment in if validating on only uncensored individuals
        # valid = val.loc[val['event'] == 0]

        train_survival_times = train.values[:,0]
        train_censored = train.values[:,1]
        train_covs = train.values[:,2:]

        valid_survival_times = valid.values[:,0]
        valid_censored = valid.values[:,1]
        valid_covs = valid.values[:,2:]

        test_survival_times = test.values[:,0]
        test_censored = test.values[:,1]
        test_covs = test.values[:,2:]

        train_loader = data.DataLoader(SurvivalDataset(train_covs, train_censored, train_survival_times), batch_size = batch_size, shuffle = True)
        validation_loader = data.DataLoader(SurvivalDataset(valid_covs, valid_censored, valid_survival_times), batch_size = len(valid_covs))
        test_loader = data.DataLoader(SurvivalDataset(test_covs, test_censored, test_survival_times), batch_size = len(test_covs))

        train_mean = torch.tensor(train_covs.mean(axis=0)).float()
        train_std = torch.tensor(train_covs.std(axis = 0)).float()

        t = 86
        cov_size = 52

    return train_loader, validation_loader, test_loader, train_mean, train_std, t, cov_size


def rps_loss(out, s_time, censored, weighting = 1, t, norm):
    '''Impelement brier loss
    Incorporate normalization, censored individuals, and potential reweighting due to time to event skew'''
    losses = 0
    for j in range(len(out)):
        #Do we want to normalize by length of time horizon? If so, reformulate what we divide each component by
        if norm:
            normed = censored[j]*(s_time[j]) + (1-censored[j]) * t
        else:
            normed = 1 
        outt = out[j]
        #Reweight due to right skew for our experiments. 
        #Large weight on early time-points was sufficient to prevent degenerate results due to event time skew
        #Other more complicated methods that resulted in similar performance included: Smaller weights up until time of event/censoring (decreasing as you get until event time)
        losses += weighting*((out[j][0] - 0)**2)
        #Following DRSA: Create surivval probabiltiesi through hazard ratios
        survs = torch.cumprod(1-outt, 0)
        #Up until event time -- ensure probability of survival closet o 1
        for i in range(s_time[j] + censored[j]):
            losses += ((1 - survs[i])**2)/normed
        #For those who are not censored, ensure survival probabliities are lower after observed event time
        if not censored[j]:
            for i in range(s_time[j], t):
                losses += (survs[i]**2)/normed
    return losses
