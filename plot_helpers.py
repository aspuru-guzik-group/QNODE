import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from qutip import *
import imageio
plt.rcParams['axes.labelsize'] = 16
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def animate_recon(xt, xm, xe, title=''):
    """x is [ts,3]"""
    images = []
    for x, label, col in zip([xt, xm, xe],['training dynamics', 'latent neural ode reconstruction','latent neural ode extrapolation' ], ['black','limegreen', 'blue']):
        for i, v in enumerate(x):
            bloch = bloch_format(Bloch())
            bloch.add_vectors(v)
            bloch.vector_color =[col]
            bloch.render()
            s = x[:i+1]
            #print(v, s[-1])
            bloch.axes.plot(s[:,1], -s[:,0], s[:,2], color=col)
            if label =='latent neural ode reconstruction':
                bloch.axes.plot(xt[:,1], -xt[:,0], xt[:,2], color='black')
            if label =='latent neural ode extrapolation':
                bloch.axes.plot(xt[:,1], -xt[:,0], xt[:,2], color='black')
                bloch.axes.plot(xm[:,1], -xm[:,0], xm[:,2], color='limegreen')

            plt.suptitle(label, fontdict={'color':col})
            plt.savefig('exp/temp_file.png')
            images.append(imageio.imread('exp/temp_file.png'))
    imageio.mimsave('exp/'+title+'.gif', images, duration=0.05)

def plot_bloch_vectors(xm, title=''):
    # xm is np x 3
    bloch = bloch_format(Bloch())
    for i, vm in enumerate(xm):
        bloch.add_vectors(vm)
        bloch.vector_color =['black']
    bloch.render()
    plt.suptitle(r'interpolated initial states $|\Psi_0 \rangle $')
    plt.savefig('exp/bvecs'+title+'.pdf', bbox_inches='tight')

def animate_traj(xt, title=''):
    """xt, xm is [ts,3] --> generate gif of both simultaneously"""
    images = []
    for i, vt in enumerate(xt):
        bloch = bloch_format(Bloch())
        bloch.add_vectors(vt)
        bloch.vector_color =['black']
        bloch.render()
        t = xt[:i+1]
        bloch.axes.plot(t[:,1], -t[:,0], t[:,2], color='black', label='dynamics')
        #plt.legend(loc='lower center')
        #plt.suptitle('latent neural ode --', fontdict={'color':'limegreen'})
        #plt.title('True quantum dynamics', fontdict={'color':'black'})
        plt.savefig('exp/temp_file.png', bbox_inches='tight')
        images.append(imageio.imread('exp/temp_file.png'))
    imageio.mimsave('exp/'+title+'.gif', images, duration=0.05)

def animate_recon_(xt, xm, title=''):
    """xt, xm is [ts,3] --> generate gif of both simultaneously"""
    images = []
    for i, (vt, vm) in enumerate(zip(xt,xm)):
        bloch = bloch_format(Bloch())
        bloch.add_vectors(vt)
        bloch.add_vectors(vm)
        bloch.vector_color =['black', 'limegreen']
        bloch.render()
        t = xt[:i+1]
        m = xm[:i+1]
        bloch.axes.plot(t[:,1], -t[:,0], t[:,2], color='black', label='train')
        bloch.axes.plot(m[:,1], -m[:,0], m[:,2], color='limegreen', label='neural ode')
        #plt.legend(loc='lower center')
        plt.suptitle('latent neural ode --', fontdict={'color':'limegreen'})
        plt.title('True quantum dynamics', fontdict={'color':'black'})
        plt.savefig('exp/temp_file.png')
        images.append(imageio.imread('exp/temp_file.png'))
    imageio.mimsave('exp/'+title+'.gif', images, duration=0.05)

def animate_single_traj(x, title=''):
    """x is [ts,3]"""
    images = []
    for i, v in enumerate(x):
        bloch = Bloch()
        bloch.add_vectors(v)
        bloch.add_points(v)
        bloch.render()
        s = x[:i+1]
        print(v, s[-1])
        bloch.axes.plot(s[:,1], -s[:,0], s[:,2], color='limegreen')
        plt.savefig('exp/temp_file.png')
        images.append(imageio.imread('exp/temp_file.png'))
    imageio.mimsave('exp/traj'+title+'.gif', images, duration=0.125)
    os.remove('exp/temp_file.png')

def plot_traj_bloch(x, title='', col='limegreen',view=[0,90]):
    bloch = bloch_format(Bloch(), view)#[-40,30])
    bloch.render()
    bloch.axes.plot(x[:,1], -x[:,0], x[:,2], color=col)
    plt.savefig('exp/'+title)

def construct_gif(xs, title=''):
    """ constructs a gif of stationary bloch trajectories """
    cmap = cm.get_cmap('Greens', len(xs))
    cols = cmap(range(len(xs)))
    images = []
    for i, x in enumerate(xs):
        filename='temp_file.png'
        plot_traj_bloch(x, filename)
        images.append(imageio.imread('exp/'+filename))
    imageio.mimsave('exp/'+title+'.gif', images, duration=0.5)
    os.remove('exp/temp_file.png')

def bloch_format(bloch, view=[0, 90]):
    bloch.frame_color = 'gray'
    bloch.frame_num = 6
    bloch.frame_alpha = 0.15
    bloch.sphere_alpha = 0.1
    bloch.sphere_color = 'whitesmoke'
    bloch.view = view
    bloch.ylabel = ['','']
    bloch.xlabel = ['','']
    bloch.zlabel = ['','']
    return bloch

def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0.:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def get_latent_interp(z1, z2, num_steps, linear=False):
    zs = []
    ratios = np.linspace(0, 1, num_steps)
    print(ratios)
    for ratio in ratios:
        if linear:
            v = (1.0 - ratio) * z1 + ratio * z2
        else:
            v = slerp(ratio, z1, z2)
        zs.append(v)
    return zs

def normalize(a):
	a = a - np.real(a).min()
	return a/np.abs(a).max()

def norm(s):
    s =np.sum(s**2,-1) **.5
    return s

def get_interpolate(model, data, i, j, n_steps=8):
    nts = len(data.train_time_steps)
    ts = torch.from_numpy(data.train_time_steps).float()
    x1 = data.train_expect_data[[i]]
    x2 = data.train_expect_data[[j]]
    trajs = np.concatenate((x1, x2), axis=0).reshape((2, nts, 3))
    trajs = torch.from_numpy(trajs).float()
    z0 = model.encode(trajs, ts, reconstruct=True)
    z1, z2 = z0[0,:], z0[1,:]
    zs = get_latent_interp(z1, z2, n_steps)
    return zs
    
def round_3sf(num_list):
    trimmed = []
    for num in num_list:
        trimmed.append(round(num, 3 - int(math.floor(math.log10(abs(num)))) - 1))
    return trimmed
