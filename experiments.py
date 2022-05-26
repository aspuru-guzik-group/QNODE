import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qutip import *
from plot_helpers import *
from model import load
import os
import matplotlib.ticker as mticker

train_end = 2
expo_start = 1.9732
expo_good_end = 6
np.random.seed(0)
torch.manual_seed(0)

matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20

def bs_train_and_sample(type, num):
    """
    1) Trajectories of training data and random samples on separate bloch spheres.
    2) Norm plot comparing random samples and training.
    """
    if not os.path.exists('plots/bs_train_and_sample'):
        os.makedirs('plots/bs_train_and_sample')
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
            plt.savefig('plots/bs_train_and_sample/' + type  + str(r) + '_' + job + '_bs.pdf', bbox_inches = 'tight', pad_inches = 0)
            plt.close()

            _, ax = plt.subplots()
            ax.set_aspect(aspect=1.7)
            ax.tick_params(width=6, length=8)
            plt.xticks([0, 1, 2, 3, 4, 5, 6], [], fontsize=40)
            if type == 'open':
                plt.ylim(0, 1.1)
                plt.yticks([0, 0.5, 1], [], fontsize=40)
            else:
                plt.ylim(0.5, 1.6)
                plt.yticks([0.5, 1.0, 1.5], [], fontsize=40)
        
            traj_norm = norm(np_trajs[i]) if job == 'train' else norm(xm[i])
            plt.plot(t[idxt], traj_norm[idxt], c=t_col, linewidth=1.3)
            plt.plot(t[idxe], traj_norm[idxe], c=e_col, linewidth=1.3)
            plt.savefig('plots/bs_train_and_sample/' + type + str(r) + '_' + job + '_norm.pdf', bbox_inches = 'tight', pad_inches = 0)
            plt.close()

def reconstruct_mse(type):
    """
    Plotting trajectories in a range of target MSEs. 
    We picked the best mse, avg mse, MSE = 0.0015 and MSE = 0.0020
    """
    if not os.path.exists('plots/recon_mse'):
        os.makedirs('plots/recon_mse')
    data, model = load(type)
    ts = torch.from_numpy(data.total_time_steps).float()
    ts_t = torch.from_numpy(data.train_time_steps).float()
    trajs = torch.from_numpy(data.train_expect_data).float()
    total_trajs = torch.from_numpy(data.total_expect_data).float()
    idxe = np.where(np.logical_and(data.total_time_steps >= expo_start, data.total_time_steps <= expo_good_end))[0]
    idxt = np.where(data.total_time_steps <= train_end)[0]
    z = model.encode(trajs, ts_t)
    xs = model.decode(z, ts)

    avg_mse, mse_errors = model.MSE(trajs, ts_t)
    idxs = np.argsort(mse_errors)
    
    target_mse = [avg_mse, 0.0015, 0.0020]
    target_mse_idxs = []
    j = 0
    for i, idx in enumerate(idxs):  
        if mse_errors[idx] >= target_mse[j]:
            target_mse_idxs.append(i)
            j += 1
        
        if j == len(target_mse):
            break

    plot_idxs = np.concatenate((idxs[:10], idxs[target_mse_idxs[0]-5:target_mse_idxs[0]+5], 
    idxs[target_mse_idxs[1]-5:target_mse_idxs[1]+5], idxs[target_mse_idxs[2]-5:target_mse_idxs[2]+5]), axis=0)
    
    for i in range(plot_idxs.shape[0]):
        t_recon = xs[plot_idxs[i]][idxt]
        e_recon = xs[plot_idxs[i]][idxe]
        t_train = total_trajs[plot_idxs[i]][idxt]
        e_train = total_trajs[plot_idxs[i]][idxe]
        
        if i < 10:
            label = 'best'
        elif i // 10 == 1:
            label = 'avg_mse'
        elif i // 10 == 2:
            label = 'mid'
        else: 
            label = 'worst'

        print('type: {}, traj: {}, mse: {:9f}'.format(type, plot_idxs[i], mse_errors[plot_idxs[i]]))
        
        _, axs = plt.subplots(3, 1)

        for j, ax in enumerate(axs):
            ax.plot(ts[idxt], t_train[:,j], c='black')
            ax.plot(ts[idxe], e_train[:,j], c='red')
            ax.plot(ts[idxt], t_recon[:, j], c='limegreen')
            ax.plot(ts[idxe], e_recon[:,j], c='blue')
            ax.set_xticks([])
            ax.set_yticks([-1, 0, 1])

            if j == len(axs) - 1:
                ax.set_xticks([0, 2, 4, 6])
                ax.set_xlabel('time(arb. units)', fontsize=25)
                

        plt.savefig('plots/recon_mse/{}_{}_mseindex_{}.pdf'.format(type, label, plot_idxs[i]), bbox_inches = 'tight', pad_inches = 0)
        plt.close()

def average_mse(type):
    """
    Plotting average MSE over time.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

    data, model = load(type)
    ts = torch.from_numpy(data.total_time_steps).float()
    ts_t = torch.from_numpy(data.train_time_steps).float()
    trajs = torch.from_numpy(data.train_expect_data).float()
    idxt = np.where(data.total_time_steps <= train_end)[0]
    idxe = np.where(np.logical_and(data.total_time_steps >= expo_start, data.total_time_steps <= expo_good_end))[0]

    multiplier = 100

    padding = 0.02 * multiplier if type == 'closed' else 0.001 * multiplier
    
    z = model.encode(trajs, ts_t)
    xs = model.decode(z, ts).numpy()
    mse = np.mean((data.total_expect_data - xs)**2, axis=0)
    mse = mse[np.where(data.total_time_steps <= expo_good_end)[0]] * multiplier
    max_mse = np.max(mse, axis=0)
    max_mse = round_3sf(max_mse)
    ts = data.total_time_steps

    _, axs = plt.subplots(3, 1)
    plt.subplots_adjust(hspace=0.3)
    
    for i, ax in enumerate(axs):
        ax.plot(ts[idxt], mse[idxt,i], c='limegreen')
        ax.plot(ts[idxe], mse[idxe,i], c='blue')
        ax.set_ylim(-padding, max_mse[i] + padding)
        ax.set_yticks([0, max_mse[i] / 2, max_mse[i]])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.set_xticks([])
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')

        if i == len(axs) - 1:
                ax.set_xticks([0, 2, 4, 6])
                ax.set_xlabel('time(arb. units)', fontsize=22)

    plt.savefig('plots/{}_avg_amp_over_mse.pdf'.format(type), bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

def mse_hist(type):
    """
    Plotting histogram of MSEs.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

    data, model = load(type)
    ts_t = torch.from_numpy(data.train_time_steps).float()
    trajs = torch.from_numpy(data.train_expect_data).float()
    _, mse_errors = model.MSE(trajs, ts_t)
    mse_errors = mse_errors * 1000

    if type == 'closed':
        target_mses = np.asarray([0.000067, 0.000809, 0.001506, 0.001928]) * 1000
    elif type == 'two':
        target_mses = np.asarray([0.000336, 0.001199, 0.001487, 0.001996]) * 1000
    else: 
        target_mses = np.asarray([0.000140, 0.001133, 0.001502, 0.001991]) * 1000

    plt.hist(mse_errors, bins=100)

    colors = ['red', 'green', 'purple', 'magenta']

    for i in range(4):
        plt.axvline(target_mses[i], c=colors[i], linewidth=1.2)
        
    plt.xlabel('Training trajectory MSE values ($10^{-2}$)', fontsize=20)
    plt.xlim([0, 6])
    plt.savefig('plots/{}_mse_hist.pdf'.format(type), bbox_inches = 'tight', pad_inches = 0)
    plt.close()

if __name__ == "__main__":
    bs_train_and_sample('closed', 100)
    bs_train_and_sample('open', 100)
    reconstruct_mse('closed')
    reconstruct_mse('open')
    average_mse('closed')
    average_mse('open')
    mse_hist('closed')
    mse_hist('open')
    mse_hist('two')
        
        



