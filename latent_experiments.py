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
    if not os.path.exists('plots/{}_interpolate'.format(type)):
        os.makedirs('plots/{}_interpolate'.format(type))
    data, model = load(type)
    idxe = np.where(np.logical_and(data.total_time_steps >= expo_start, data.total_time_steps <= time_end))[0]
    idxt = np.where(data.total_time_steps <= train_end)[0]
    ts = data.total_time_steps[np.where(data.total_time_steps <= time_end)]
    ts = torch.from_numpy(ts).float()
    t = ts.numpy()
    idx = np.random.randint(0,data.total_expect_data.shape[0], size=(n))
    idx_ = np.random.randint(0,data.total_expect_data.shape[0], size=(n))
    
    if type == 'closed':
        idx = [465]
        idx_ = [694]
    else:
        idx = [36]
        idx_ = [124]

    
    test = zip(idx,idx_)

    for i, j in test:
        zs = get_interpolate(model, data, i,j, n_steps=n_steps)
        xs = []
        title = type + '_interpolate_from_' +str(i)+ '-' +str(j)

        for l, z in enumerate(zs):
            view = [-50, 30]
            bloch = bloch_format(Bloch(), view)
            bloch.render()
            x = model.decode(z, ts).numpy()
            bloch.axes.plot(x[:,1][idxt], -x[:,0][idxt], x[:,2][idxt], color='limegreen')
            bloch.axes.plot(x[:,1][idxe], -x[:,0][idxe], x[:,2][idxe], color='blue')
            xs.append(x)
            plt.savefig('plots/{}_interpolate/{}-{}.pdf'.format(type, title, l), bbox_inches = 'tight', pad_inches = 0)
            plt.close()

        construct_gif(xs, type, title)

        #plot norm 
        for k, traj in enumerate(xs):
            fig, ax = plt.subplots()
            if type == 'closed':
                plt.ylim(0.5, 1.6)
                plt.yticks([0.5, 1.0, 1.5], [], fontsize=28)
            else:
                plt.ylim(0, 1.1)
                plt.yticks([0, 0.5, 1.0], [], fontsize=28)
            
            plt.xticks([0, 1, 2, 3, 4, 5, 6], [], fontsize=28)
            ax.set_aspect(aspect=1.7)
            ax.tick_params(width=6, length=8)
            traj_norm = norm(traj)
            plt.plot(t[idxt], traj_norm[idxt], c='limegreen')
            plt.plot(t[idxe], traj_norm[idxe], c='blue')
            plt.savefig('plots/{}_interpolate/{}_norm-{}.pdf'.format(type, title, k), bbox_inches = 'tight', pad_inches = 0)
            plt.close()


        #plot latent dynamics
        zs_torch = torch.FloatTensor(len(zs), zs[0].shape[0])
        torch.cat(zs, out=zs_torch)
        zs_torch = torch.reshape(zs_torch, (len(zs), zs[0].shape[0]))
        zts = odeint(model.func, zs_torch, ts).permute(1, 0, 2)
        zts = zts.detach().numpy()

        if type == 'closed':
            azim = 135
            elev = 30
        else:
            azim = 90
            elev = 8
        
        for p, x in enumerate(zts):
            fig = plt.figure()
            axes = Axes3D(fig, azim=azim, elev=elev, auto_add_to_figure=False)
            fig.add_axes(axes)
            axes.plot(x[:,0][idxt], x[:,1][idxt], x[:,2][idxt], color='limegreen', alpha=0.5)
            axes.plot(x[:,0][idxe], x[:,1][idxe], x[:,2][idxe], color='blue', alpha=0.5)
            plt.yticks([])
            plt.xticks([])
            axes.set_zticks([])
            plt.savefig('plots/{}_interpolate/{}_ld-{}.pdf'.format(type, title, p), bbox_inches = 'tight', pad_inches = 0)
            plt.close()


if __name__ == "__main__":
    interpolate('closed', 1)
    interpolate('open', 1)