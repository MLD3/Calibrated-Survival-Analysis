import numpy as np
import pandas as pd
import torch as torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils import data
from sklearn.model_selection import train_test_split
import pickle as pickle
import bisect
from collections import Counter
import argparse
import os
import shutil
from utils import *

parser = argparse.ArgumentParser(description='Process model parameters.')

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Train DRSA.")

    parser.add_argument('--dataset', type=str, required=False, default='CLINIC',help='Which dataset to use?')

    parser.add_argument('--use_log', type=bool, required= False, default = False,
                        help='Whether to use logarithmic loss')

    parser.add_argument('--use_brier', type=bool, required=False, default = True,
                        help='Whether to use Brier score L_RPS loss')

    parser.add_argument('--lambda', type=int, required=False, default = 1,
                        help='Control the trade-off between L_RPS and L_kernel')

    parser.add_argument('--norm_mse', type=bool, required=False, default=False,
                        help='Whether to normalize MSE by time-points')

    parser.add_argument('--sigma', type=float, required=False,  default = -1,
                        help='Whether to use kernel loss L_kernel. Default is -1, which means not to use it')

    parser.add_argument('--n_iters', type=int, required=False, default= 5,
                        help='Number of iterations with different weight initializations to run')

    parser.add_argument('--hidden_size', type=int, required=False, default = 100,
                        help='Hiddden Size of LSTM')

    parser.add_argument('--batch_size', type=int, required=False, default = 50,
                        help='Batch Siize')

    parser.add_argument('--learning_rate', type=float, required=False, default = 1e-3,
                        help='Learning Rate')

    parser.add_argument('--output_name', type=str, required=False, default = 'results.pkl',
                        help='pkl name to output results')

    parser.add_argument('--model_name', type=str, required=False, default = 'model/drsa',
                        help='Model output name')
    
    return parser.parse_args()





#Model architecture 
class DRSA(nn.Module):

    def __init__(self, input_dim, hidden_size, t, num_layers = 1, bias=True, use_gpu = False):
        super(DRSA, self).__init__()

        '''
        do_bn: if set to True, then do BatchNormalization
        '''

        if use_gpu:
            self.float = torch.cuda.FloatTensor
        else:
            self.float = torch.FloatTensor
        self.cov_size = input_dim
        self.t = t
        self.lstm = nn.LSTM(input_dim, hidden_size = hidden_size, batch_first = True)  
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.t = t
        self.input_dim = input_dim
        self.sig = nn.Sigmoid()
        self.num_layers = num_layers
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.float), requires_grad = False)
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.float), requires_grad = False)
        outs, hidden = self.lstm(x, (h0,c0))
        outs = outs.contiguous().view(batch_size * self.t, self.hidden_size)
        x = self.sig(self.fc(outs)).view(batch_size, self.t)
        return x



def save_checkpoint(state, is_best, save_filename, best_save_filename):
    torch.save(state, save_filename)
    if is_best:
        shutil.copyfile(save_filename, best_save_filename)

def load_checkpoint(load_filename):
    print("=> loading checkpoint '{}'".format(load_filename))
    checkpoint = torch.load(load_filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_loss = checkpoint['best_loss']
    print("=> loaded checkpoint '{}' (iteration {})"
          .format(load_filename, checkpoint['niter']))
    print('best validation loss {}'.format(best_loss))

#Main train loop, trains models with specific paramters that can be adjusted\
def train(dataset, hidden_size, do_log, do_brier, sigma, n_iters, batch_size, output_name, model_name, lamb, norm_mse, lr):
    if os.path.isfile(output_name):
        with open(output_name,'rb') as f:
            past_results = pickle.load(f)
        full_cs,full_ddcs = past_results['Cs'], past_results['DDCs'] 
    else:
        full_cs,full_ddcs = {},{}
    full_cs[(do_log,do_brier,sigma)] = []
    full_ddcs[(do_log,do_brier,sigma)] = []
    if not do_brier and not do_log:
        #Situation with only kernel loss
        only_do_l2 = True
    if sigma < 0:
        #No kernel loss
        do_l2 = 0
    else:
        #kernel loss with specific lambda value 
        do_l2 = lamb

    #Repeat training, validation, test process multiple times to evaluate stability
    for repetition in range(n_iters):
        train_loader, validation_loader, test_loader, train_mean, train_std, t, cov_size = return_split(dataset,batch_size,0)
        net = DRSA(cov_size, hidden_size, t)
        optimizer = torch.optim.Adam(net.parameters(), lr = lr)
        save_filename = model_name + str(repetition) + '.pth.tar'
        best_save_filename = model_name + str(repetition) + '_best.pth.tar'
        val_stopping_criterion = 1e-4
        best_iter = 0
        #100 epochs was enough for convergence for our experiments, can increase if necessary for particular datasets
        epochs = 100
        i=0
        best_val_loss = -np.inf
        done = False
        for epoch in range(epochs):
            for cov, cens, s_time in train_loader:
                optimizer.zero_grad()
                s_time = s_time.long()
                cov = (cov.float() - train_mean) / train_std
                cov = cov.repeat(1, t).view(-1, t, cov_size-1)
                time = torch.arange(1, t+1, 1).repeat(cov.size(0)).view(-1, t, 1).float()
                cov = torch.cat((cov, time), dim = 2)
                out = net.forward(cov)
                surv_probs = torch.cumprod(1 - out  * torch.cat([torch.zeros(1), torch.ones(t-1)]), dim = 1)
                cens = cens.float()
                #Gather hazard ratios and log probabilities around event time
                #Estimated Survival probability at time one before observed event time
                one_before_time = surv_probs.gather(1, (s_time-1).view(-1,1)).view(-1)
                #Hazard ratio at time of observed event
                out_at_time = out.gather(1, (s_time).view(-1,1)).view(-1)
                #Estimated survival probabilities at observed event times
                at_times = surv_probs.gather(1,(s_time).view(-1,1)).view(-1)
                #Estimated survival probabilities at final time-point
                at_end = surv_probs.gather(1, torch.tensor(t-1).repeat(1, cov.size(0)).view(cov.size(0), 1)).view(-1)
                
                #All uncensored indivs should have low survival probability at end
                y_uncensored = -sum(torch.log(1 - at_end).view(-1) * (1-cens))

                #All censored individuals should have high estimated times at censoring time
                y_censored = -sum(torch.log(at_times).view(-1) * cens)

                #High hazard ratio for time of event for uncensored individuals 

                uncensored_at_times = -sum((torch.log(out_at_time * one_before_time)).view(-1) * (1-cens))
                
                f = surv_probs
                l2 = 0
                #Impelement kernel loss
                if do_l2:
                    for subj in range(len(s_time)):
                        #For all individuals with later observed event times
                        for index in (s_time > s_time[subj]).nonzero().reshape(-1): 
                            #For stability, focus only on those who are not censored. Can incorporate censored individuals here
                            if not cens[subj] and not cens[index]:
                                #Penalize discordant rankings
                                l2 += torch.exp(-(((1-f[subj][s_time[subj]]) - (1-f[index][s_time[subj]]))/sigma))
                if do_brier:
                    #RPS + Kernel
                    loss = rps_loss(out, s_time, cens.long(), weighting, t, norm_mse) + do_l2*(l2) 
                elif only_do_l2: 
                    #Kernel
                    loss = l2
                else:
                    #Log likelihood 
                    loss = y_uncensored + uncensored_at_times + do_l2*(l2) + y_censored
                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    #Repeat same steps as above for validation set
                    for cov, cens, s_time in validation_loader:
                        net.eval()
                        s_time = s_time.long()
                        cov = (cov.float() - train_mean) / train_std
                        cov = cov.repeat(1, t).view(-1, t, cov_size-1)
                        time = torch.arange(1, t+1, 1).repeat(cov.size(0)).view(-1, t, 1).float()
                        cov = torch.cat((cov, time), dim = 2)
                        out = net.forward(cov).detach()
                        surv_probs = torch.cumprod(1 - out * torch.cat([torch.zeros(1), torch.ones(t-1)]), dim = 1)
                        cens = cens.float()
                        one_before_time = surv_probs.gather(1, (s_time-1).view(-1,1)).view(-1)
                        out_at_time = out.gather(1, (s_time).view(-1,1)).view(-1)
                        at_times = surv_probs.gather(1,(s_time).view(-1,1)).view(-1)
                        at_end = surv_probs.gather(1, torch.tensor(t-1).repeat(1, cov.size(0)).view(cov.size(0), 1)).view(-1)
                        y_uncensored = -sum(torch.log(1 - at_end).view(-1) * (1-cens))
                        y_censored = -sum(torch.log(at_times).view(-1) * cens)
                        uncensored_at_times = -sum((torch.log(out_at_time * one_before_time)).view(-1) * (1-cens))
                        f = surv_probs
                        l2 = 0
                        if do_l2:
                            for subj in range(len(s_time)):
                                for index in (s_time > s_time[subj]).nonzero().reshape(-1): 
                                    if not cens[subj] and not cens[index]:
                                        l2 += torch.exp(-(((1-f[subj][s_time[subj]]) - (1-f[index][s_time[subj]]))/sigma))
                        #We choose best val loss as the one that is highest, so take negative of the loss
                        if do_brier:
                            val_loss = -(rps_loss(out, s_time, cens.long(), weighting, t) + do_l2*(l2)) 
                        elif only_do_l2: 
                            val_loss = -l2   
                        else:
                            val_loss = -(y_uncensored + uncensored_at_times + do_l2*(l2) + censored_log*y_censored)

                    is_best = False
                    #Save every X timepoint, best model achieved when the val loss is no longer decreasing
                    if ((val_loss - best_val_loss) > val_stopping_criterion):
                        is_best = True
                        best_val_loss = val_loss
                        best_iter = i
                    save_checkpoint({
                            'niter': i + 1,
                            'arch': str(type(net)),
                            'state_dict': net.state_dict(),
                            'best_loss': best_val_loss,
                            'optimizer': optimizer.state_dict(),
                        }, is_best, save_filename, best_save_filename)                
                    net.train()

                i+=1

        load_checkpoint(best_save_filename)

        #Evaluate on test set -- should also evaluate on validation set for hyperparamter selection
        surv_curvs = []
        s_times = []
        censoreds = []
        num = 0
        denom = 0

        for cov, cens, s_time in test_loader:
            net.eval()
            s_time = s_time.long()
            cov = (cov.float() - train_mean) / train_std
            cov = cov.repeat(1, t).view(-1, t, cov_size-1)
            time = torch.arange(1, t+1, 1).repeat(cov.size(0)).view(-1, t, 1).float()
            cov = torch.cat((cov, time), dim = 2)
            out = net.forward(cov).detach() * torch.cat([torch.zeros(1), torch.ones(t-1)])
            cens = cens.float()
            censoreds += list(cens.numpy())
            f = torch.cumprod(1 - out, dim = 1)
            s_times += (list(s_time.numpy()))
            surv_curvs += list(f.numpy())

            #Calculate C-Index
            for subj in range(len(s_time)):
                #If people have the same event time, can incorporate into C-Index ranking by assigning 0.5 score, or can be ignored
                for index in (s_time > s_time[subj]).nonzero().reshape(-1): 
                    #Penalize discordant pairs -- only count those that are conconrdant with observed event times
                    if not cens[subj]:
                        num += int(1-f[subj][s_time[subj]] > 1-f[index][s_time[subj]])
                        denom += 1
        full_cs[(do_log,do_brier,sigma)].append(num/denom)


        #Bin estimated surv. probabilities at obesrved event times between 0 and 1    
        binned = binned_dist(surv_curvs, s_times, censoreds, 10)

        full_ddcs[(do_log,do_brier,sigma)].append(ddc(np.array(binned)/sum(binned), np.ones(10)*.1))

        full_dct = {'Cs': full_cs, 'DDCs': full_ddcs}

        with open(output_name,'wb') as f:
            pickle.dump(full_dct, f)                                
  
if __name__ == "__main__":
    args = parse_args()
    train(args.dataset, args.hidden_size, args.use_log, args.use_brier, args.sigma, args.n_iters, args.batch_size, args.output_name, args.model_name, args.lambda, args.norm_mse, args.learning_rate)