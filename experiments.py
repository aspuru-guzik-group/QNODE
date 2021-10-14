import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from qutip import *
from plot_helpers import *
import imageio
from model import load
from sklearn.decomposition import PCA

train_end = 2
expo_start = 1.9732
expo_good_end = 4
np.random.seed(0)
torch.manual_seed(0)

def bs_train_and_sample(type, num):
    expo_good_end = 6
    data, model = load(type)
    n = data.total_expect_data.shape[0]
    ts = torch.from_numpy(data.total_time_steps).float()
    idxe = np.where(np.logical_and(data.total_time_steps >= expo_start, data.total_time_steps <= expo_good_end))[0]
    idxt = np.where(data.total_time_steps <= train_end)[0]
    rands = np.random.randint(0, n, num)
    np_trajs = data.total_expect_data[rands]
    trajs = torch.from_numpy(np_trajs).float()
    xm = model.decode(torch.randn(num, 6), ts).numpy()
    t= ts.numpy()
    view = [-50, 30]
    
    for i in range(num):
        r = rands[i]
        t_samples = xm[i][idxt]
        e_samples = xm[i][idxe]
        t_train = trajs[i][idxt]
        e_train = trajs[i][idxe]
        jobs = ['train', 'sample']
        for job in jobs:
            if job == 'train':
                t_trajs = t_train
                e_trajs = e_train
                t_col = 'black'
                e_col = 'red'
            else:
                t_trajs = t_samples
                e_trajs = e_samples
                t_col = 'limegreen'
                e_col = 'blue'
            
            bloch = bloch_format(Bloch(), view)
            bloch.render()
            bloch.axes.plot(t_trajs[:,0], t_trajs[:,1], t_trajs[:,2], c=t_col)
            bloch.axes.plot(e_trajs[:,0], e_trajs[:,1], e_trajs[:,2], c=e_col)
            plt.savefig('plots/bs_train_and_sample/' + type  + str(r) + '_' + job + '_bs.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()

            fig, ax = plt.subplots()
            ax.set_aspect(aspect=1.7)
            if type == 'open':
                plt.ylim(0, 1)
                plt.yticks([0, 1], fontsize=28)
            else:
                plt.ylim(0.5, 1.5)
                plt.yticks([0.5, 1.0, 1.5], fontsize=28)
        
            plt.xticks([0, 6], fontsize=28)

            traj_norm = norm(np_trajs[i]) if job == 'train' else norm(xm[i])
            plt.plot(t[idxt], traj_norm[idxt], c=t_col, linewidth=1.3)
            plt.plot(t[idxe], traj_norm[idxe], c=e_col, linewidth=1.3)
            plt.savefig('plots/bs_train_and_sample/' + type + str(r) + '_' + job + '_norm.png', bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        
if __name__ == "__main__":
    bs_train_and_sample('closed', 1)
    bs_train_and_sample('open', 1)
        



