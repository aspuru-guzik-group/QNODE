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
from plot_helpers import norm
from torchdiffeq import odeint

train_end = 2
expo_start = 1.9732
expo_good_end = 4
np.random.seed(0)
torch.manual_seed(0)

def plot_traj_bloch(x, type='closed', title='', col='limegreen'):
    view = [-50, 30]
    bloch = bloch_format(Bloch(), view)
    bloch.render()
    bloch.axes.plot(x[:,0], x[:,1], x[:,2], color=col)
    plt.savefig('plots/{}_interpolate/{}'.format(type, title), bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def construct_gif(xs, type='closed', title='', cols=[], cmap='viridis'):
    """ constructs a gif of stationary bloch trajectories """
    if len(cols) == 0:
        cmap = cm.get_cmap(cmap, len(xs))
        cols = cmap(range(len(xs)))
    images = []
    for i, x in enumerate(xs):
        filename='temp_file.png'
        plot_traj_bloch(x, type, filename, col=cols[i])
        images.append(imageio.imread('plots/{}_interpolate/{}'.format(type, filename)))
    imageio.mimsave('plots/{}_interpolate/{}.gif'.format(type, title), images, duration=0.5)
    os.remove('plots/{}_interpolate/{}'.format(type, filename))

def interpolate(type='closed', n=25, n_steps=8, time_end=6):
    data, model = load(type)
    idxe = np.where(np.logical_and(data.total_time_steps >= expo_start, data.total_time_steps <= time_end))[0]
    idxt = np.where(data.total_time_steps <= train_end)[0]
    ts = data.total_time_steps[np.where(data.total_time_steps <= time_end)]
    ts = torch.from_numpy(ts).float()
    t = ts.numpy()
    idx = np.random.randint(0,data.total_expect_data.shape[0], size=(n))
    idx_ = np.random.randint(0,data.total_expect_data.shape[0], size=(n))
    test = zip(idx,idx_)
    
    for i, j in test:
        zs = get_interpolate(model, data, i,j, n_steps=n_steps)
        xs = []
        title = type + ' interpolate from ' +str(i)+ '-' +str(j)

        for l, z in enumerate(zs):
            view = [-50, 30]
            bloch = bloch_format(Bloch(), view)
            bloch.render()
            x = model.decode(z, ts).numpy()
            bloch.axes.plot(x[:,1][idxt], -x[:,0][idxt], x[:,2][idxt], color='limegreen')
            bloch.axes.plot(x[:,1][idxe], -x[:,0][idxe], x[:,2][idxe], color='blue')
            xs.append(x)
            plt.savefig('plots/{}_interpolate/{}-{}.png'.format(type, title, l), bbox_inches = 'tight', pad_inches = 0)
            plt.close()

        construct_gif(xs, type, title)

        #plot norm 
        for k, traj in enumerate(xs):
            fig, ax = plt.subplots()
            plt.ylim(0, 1)
            plt.yticks([0, 1], fontsize=28)
            plt.xticks([0, 6], fontsize=28)
            ax.set_aspect(aspect=1.7)
            traj_norm = norm(traj)
            plt.plot(t[idxt], traj_norm[idxt], c='limegreen')
            plt.plot(t[idxe], traj_norm[idxe], c='blue')
            plt.savefig('plots/{}_interpolate/{}_norm-{}.png'.format(type, title, k), bbox_inches = 'tight', pad_inches = 0)
            plt.close()


        #plot latent dynamics
        zs_torch = torch.FloatTensor(len(zs), zs[0].shape[0])
        torch.cat(zs, out=zs_torch)
        zs_torch = torch.reshape(zs_torch, (len(zs), zs[0].shape[0]))
        zts = odeint(model.func, zs_torch, ts).permute(1, 0, 2)
        zts = zts.detach().numpy()
        
        for p, x in enumerate(zts):
            fig = plt.figure()
            axes = Axes3D(fig, azim=90, elev=8, auto_add_to_figure=False)
            fig.add_axes(axes)
            axes.plot(x[:,0][idxt], x[:,1][idxt], x[:,2][idxt], color='limegreen', alpha=0.5)
            axes.plot(x[:,0][idxe], x[:,1][idxe], x[:,2][idxe], color='blue', alpha=0.5)
            plt.yticks([])
            plt.xticks([])
            axes.set_zticks([])
            plt.savefig('plots/{}_interpolate/{}_ld-{}.png'.format(type, title, p), bbox_inches = 'tight', pad_inches = 0)
            plt.close()


if __name__ == "__main__":
    interpolate('closed', 1)
    interpolate('open', 1)