import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from qutip import *
from plot_helpers import *
from model import load
import matplotlib.ticker as mticker

train_end = 2
expo_start = 1.9732
expo_good_end = 6
np.random.seed(0)
torch.manual_seed(0)

matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20

def train_and_sample(num):
    """
    Plotting random samples and training data over time.
    """
    if not os.path.exists('plots/two_qubit_train_and_sample'):
        os.makedirs('plots/two_qubit_train_and_sample')
    data, model = load('two')
    n = data.total_expect_data.shape[0]
    t = data.total_time_steps
    ts = torch.from_numpy(t).float()
    idxt = np.where(data.total_time_steps <= train_end)[0]
    xm = model.decode(torch.randn(num, 8), ts).numpy()
    train_data = data.train_expect_data
    
    for i in range(num):
        t_samples = xm[i][idxt]
        cloned = np.tile(t_samples, (n, 1, 1))
        mse = np.mean((train_data - cloned) ** 2, axis=1)
        mse = np.mean(mse, axis=1)
        target_traj = train_data[np.argmin(mse)]
            
        _, axs = plt.subplots(4, 1)
        plt.subplots_adjust(hspace=0.3)

        for j, ax in enumerate(axs):
            ax.plot(ts[idxt], target_traj[:, j], c='black')
            ax.set_yticks([-1, 0, 1])

            if j == len(axs) - 1:
                ax.set_xticks([0, 1, 2])
            else:
                ax.set_xticks([])

        plt.savefig(f'plots/two_qubit_train_and_sample/train_data_{i}.pdf', bbox_inches = 'tight', pad_inches = 0)
        plt.close()

        _, axs = plt.subplots(4, 1)
        plt.subplots_adjust(hspace=0.3)

        for k, ax in enumerate(axs):
            ax.plot(ts[idxt], t_samples[:, k], c='limegreen')
            ax.set_yticks([-1, 0, 1])

            if k == len(axs) - 1:
                ax.set_xticks([0, 1, 2])
            else:
                ax.set_xticks([])
        
        plt.savefig(f'plots/two_qubit_train_and_sample/sample_{i}.pdf', bbox_inches = 'tight', pad_inches = 0)
        plt.close()
        

def reconstruct_mse():
    """
    Plotting trajectories in a range of target MSEs. 
    We picked the best mse, avg mse, MSE = 0.0015 and MSE = 0.0020
    """
    if not os.path.exists('plots/two_qubit_recon_mse'):
        os.makedirs('plots/two_qubit_recon_mse')
    data, model = load('two')
    ts = torch.from_numpy(data.total_time_steps).float()
    ts_t = torch.from_numpy(data.train_time_steps).float()
    trajs = torch.from_numpy(data.train_expect_data).float()
    total_trajs = torch.from_numpy(data.total_expect_data).float()
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
        avg_amp = (np.max(trajs.numpy()[plot_idxs[i]], axis=0) - np.min(trajs.numpy()[plot_idxs[i]], axis=0)) / 2
        avg_amp = np.mean(avg_amp)
        t_recon = xs[plot_idxs[i]][idxt]
        t_train = total_trajs[plot_idxs[i]][idxt]

        print('type: two, traj: {}, mse: {:9f}'.format(plot_idxs[i], mse_errors[plot_idxs[i]]))

        if i < 10:
            label = 'best'
        elif i // 10 == 1:
            label = 'avg_mse'
        elif i // 10 == 2:
            label = 'mid'
        else: 
            label = 'worst'

        _, axs = plt.subplots(4, 1)
        plt.subplots_adjust(hspace=0.3)

        for j, ax in enumerate(axs):
            ax.plot(ts[idxt], t_recon[:, j], c='limegreen')
            ax.plot(ts[idxt], t_train[:,j], c='black')
            ax.set_yticks([-1, 0, 1])

            if j == len(axs) - 1:
                ax.set_xticks([0, 1, 2])
            else:
                ax.set_xticks([])

        plt.savefig('plots/two_qubit_recon_mse/two_{}_mseindex_{}.pdf'.format(label, plot_idxs[i]), bbox_inches = 'tight', pad_inches = 0)
        plt.close()

def average_mse():
    """
    Plotting average MSE over time
    """
    data, model = load('two')
    ts_t = torch.from_numpy(data.train_time_steps).float()
    trajs = torch.from_numpy(data.train_expect_data).float()
    z = model.encode(trajs, ts_t)
    xs = model.decode(z, ts_t).numpy()
    multiplier = 1000
    mse = np.mean((data.train_expect_data - xs)**2, axis=0) * multiplier
    max_mse = np.max(mse, axis=0)
    max_mse = round_3sf(max_mse)

    _, axs = plt.subplots(4, 1)
    plt.subplots_adjust(hspace=0.5)

    padding = 0.0005 * multiplier

    for i, ax in enumerate(axs):
        ax.plot(data.train_time_steps, mse[:, i], c='limegreen')
        ax.set_ylim(-padding, max_mse[i] + padding)
        ax.set_yticks([0, max_mse[i] / 2, max_mse[i]])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_xticks([])
        
        if i == len(axs) - 1:
                ax.set_xticks([0, 0.5, 1.0, 1.5, 2])
                ax.set_xlabel('time(s)', fontsize=20)

    plt.savefig('plots/two_avg_amp_over_mse.pdf', bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

if __name__ == "__main__":
    # train_and_sample(20)
    # reconstruct_mse()
    average_mse()