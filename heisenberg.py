import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from qutip import *
from plot_helpers import *

from model import load

train_end = 2
expo_start = 1.9732
expo_good_end = 4

def heisenberg_plot(type, samples=10, iter=0):
    if not os.path.exists('plots/{}_heisenberg'.format(type)):
        os.makedirs('plots/{}_heisenberg'.format(type))
    data, model = load(type)
    time_end = 6
    idx_ex = np.where(np.logical_and(data.total_time_steps >= expo_start, data.total_time_steps <= time_end))[0]
    idx_train = np.where(data.total_time_steps <= train_end)[0]
    time_steps = torch.from_numpy(data.total_time_steps[np.where(data.total_time_steps <= time_end)]).float()

    decoded = model.decode(torch.randn(samples, 6), time_steps).numpy()
    var_x = 1 - decoded[:,:,0] ** 2 
    var_z = 1 - decoded[:,:,2] ** 2

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # a,b = sx.min(), sx.min()
    x = np.linspace(-1/2, 1, 180)
    z = np.linspace(0, 6, 180)
    cmap = cm.get_cmap('Greens', samples)
    col_g = cmap(range(samples))
    cmap = cm.get_cmap('Blues', samples)
    col_b = cmap(range(samples))

    X, Z = np.meshgrid(x, z)
    Y = 1 - X
    ax.plot_surface(X, Y, Z, color='lightgrey', alpha=0.75)
    
    for j in range(180):
        if j <= 60:
            ax.scatter3D(var_x[:, j], var_z[:, j], j / 30, marker='.', color=col_g, alpha=0.75)
        else:
            ax.scatter3D(var_x[:, j], var_z[:, j], j / 30, marker='.', color=col_b, alpha=0.75)
    
    ax.set_xlabel(r" var$(x)  $")
    ax.set_zlabel(r" time(arb. unit) ", fontsize=12)
    ax.set_ylabel(r" var$(z) $")
    ax.view_init(elev=45, azim=-55)
    ax.text(-0.5,0.75,179, r' var$(x) + $var$(z) =1 $', rotation=90, fontsize=15, rotation_mode="anchor")
    title = "heisenberg_" + type
    plt.savefig('plots/{}_heisenberg/{}_{}-{}.png'.format(type, samples, title, iter), bbox_inches='tight')
    plt.close()

def main():
    for k in range(1):
        heisenberg_plot('closed', 50, iter=k)
        heisenberg_plot('open', 50, iter=k)
    
if __name__ == '__main__':
    main()
    
