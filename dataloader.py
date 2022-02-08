from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from torch.utils.data import Dataset
    
def normalize(a):
	a_oo = a - np.real(a).min()
	return a_oo/np.abs(a_oo).max()

def get_state(theta, phi):
    ket0, ket1 = np.array([[1.],[0.]]), np.array([[0.],[1.]])
    bloch_state = np.cos(theta/2) * ket0 + np.exp(np.complex(0, phi))*ket1
    return Qobj(bloch_state)

def get_spherical(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

def sample_bloch(n_samples=50, rand=True):
    if rand:
        thetas = np.sort(np.pi * np.random.rand(n_samples))
        phis = np.sort(2 * np.pi * np.random.rand(n_samples))
        
    else:
        thetas = np.linspace(0, np.pi, n_samples)
        phis = np.linspace(0, 2 * np.pi, n_samples)
    
    bloch_vec = np.dstack(np.meshgrid(thetas, phis)) # [n_samples, n_samples, 2]
    return bloch_vec.reshape(n_samples * n_samples, 2) # [n_samples^2, 2]

def sample_initial_states(n_samples=50, rand=True):
    " sample initial states "
    bloch_vecs = sample_bloch(n_samples, rand)
    states = [get_state(*bvec) for bvec in bloch_vecs]
    spherical = np.asarray([get_spherical(*bvec) for bvec in bloch_vecs])

    return states, bloch_vecs, spherical

def final_states_to_numpy(states):
    "convert list of quantum objects to numpy array [2, num_time_steps]"
    return np.concatenate([state.full() for state in states], axis=-1)

class StochasticTwoLevelDataset(Dataset):
    def __init__(self, num_batches=30, batched_samples=6, validation_samples=10, start=0, stop=2, last=10, time_steps=300, mc_samples=250, dataset_type='closed'): 
        self.total_time_steps = np.linspace(start, last, time_steps)
        self.initial_states, _, self.spherical = sample_initial_states(batched_samples, rand=True)
        self.validation_points = sample_initial_states(validation_samples, rand=False)
        self.num_per_batch = batched_samples ** 2
        self.num_batches = num_batches
        self.num_trajs = self.num_per_batch * self.num_batches
        self.dataset_type = dataset_type

        if dataset_type == 'closed':
            self.rand_parameters = np.zeros((num_batches, 2))
        elif dataset_type == 'open':
            self.rand_parameters = np.zeros((num_batches, 4))
        expect_data = []
        for i in range(num_batches):
            samp_z = np.random.uniform(1, 2.5, 1)
            samp_x = np.random.uniform(1, 2.5, 1)
            self.rand_parameters[i, 0] = samp_z
            self.rand_parameters[i, 1] = samp_x
            H = samp_z[0] * sigmaz() + samp_x[0] * sigmax()

            if dataset_type == 'closed':
                solve = lambda state : sesolve(H, state, self.total_time_steps, e_ops=[sigmax(), sigmay(), sigmaz()], progress_bar=None)
            elif dataset_type == 'open':
                decay_samp = np.random.uniform(0.1, 0.3, 2)
                self.rand_parameters[i, 2:] = decay_samp
                c_ops = [np.sqrt(decay_samp[0]) * sigmax(), np.sqrt(decay_samp[1]) * sigmaz()]
                solve = lambda state : mesolve(H, state, self.total_time_steps, e_ops=[sigmax(), sigmay(), sigmaz()], c_ops=c_ops)
                
            all_states = [solve(state).expect for state in self.initial_states]
            states = [np.asarray(states, dtype='double') for states in all_states] 
            states = np.asarray([np.column_stack([state[0], state[1], state[2]]) for state in states])
            expect_data.append(states)
            
        self.expect_data = np.asarray(expect_data)
        self.total_expect_data = self.expect_data.reshape(self.num_trajs, time_steps, 3)
        self.train_time_steps = self.total_time_steps[np.where(self.total_time_steps <= stop)]
        self.train_expect_data = self.total_expect_data[:,:self.train_time_steps.shape[0],:]

    def plot_trajs(self):
        for i in range(self.num_batches):
            for j in range(self.num_per_batch):
                ts = self.time_steps
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

                ax1.plot(ts, self.expect_data[i, j, :, 0])
                ax1.set_ylim(-1, 1)
                ax1.set_ylabel('$\sigma_x$')

                ax2.plot(ts, self.expect_data[i, j, :, 1])
                ax2.set_ylim(-1, 1)
                ax2.set_ylabel('$\sigma_y$')

                ax3.plot(ts, self.expect_data[i, j, :, 2])
                ax3.set_ylim(-1, 1)
                ax3.set_ylabel('$\sigma_z$')
                if self.dataset_type == 'closed':
                    ax3.set_xlabel('H = {}z + {}x'.format(self.rand_parameters[i, 0], self.rand_parameters[i, 1]))
                else:
                    ax3.set_xlabel('H = {}z + {}x decay: {} {}'.format(*self.rand_parameters[i]))

                plt.savefig('plots/stochastic_closed_noise/traj_{}_{}.png'.format(i, j))
                plt.close(fig)

    def render_initial_states(self, directory):
        bloch = Bloch()
        colors = normalize(self.spherical)
        bloch.point_color = colors
        bloch.add_points([self.spherical[:, 0], self.spherical[:, 1], self.spherical[:, 2]], 'm')
        bloch.save(directory)

# two qubit functions

def random_u(N):
    #Return a Haar distributed random unitary NxN
    #N being the system dimension
    Z = np.random.randn(N,N) + 1.0j * np.random.randn(N,N)
    [Q,R] = np.linalg.qr(Z)    # QR decomposition
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)

def random_psi():
    #Return random state, within computational subspace {|0>,|1>} 
    Ur = random_u(2)
    alpha = Ur[0,0]
    beta = Ur[1,0]
    ket0, ket1 = np.array([[1.],[0.]]), np.array([[0.],[1.]])
    rand_vector = alpha * ket0 + beta * ket1 # alpha |0> + beta |1>
    return alpha, beta, rand_vector

def two_qubit_initial(num):
    initial_states = []
    for i in range(num):
        _, _, vec1 = random_psi()
        _, _, vec2 = random_psi()
        initial_states.append(Qobj(np.kron(vec1, vec2)))
    return initial_states


class TwoQubitDataset(Dataset):
    def __init__(self, omega=1, delta=1, J=1, num_batches=30, num_trajs=36, time_steps=300, stop=2, end=10):
        sigmaz1, sigmaz2 = Qobj(np.kron(sigmaz(), np.eye(2))), Qobj(np.kron(np.eye(2), sigmaz()))
        sigmax1, sigmax2 = Qobj(np.kron(sigmax(), np.eye(2))), Qobj(np.kron(np.eye(2), sigmax()))

        self.num_trajs = num_batches * num_trajs
        self.initial_states = two_qubit_initial(num_trajs)
        self.total_time_steps = np.linspace(0, end, time_steps)

        expect_data = []
        for i in range(num_batches):
            samp_z = np.random.uniform(1, 2.5, 1)[0]
            samp_x = np.random.uniform(1, 2.5, 1)[0]
            self.H = (omega / 2 * sigmaz1 * samp_z) + (delta / 2 * sigmax1 * samp_x) + (omega / 2 * sigmaz2 * samp_z) + (delta / 2 * sigmax2 * samp_x) + (J * sigmax1 * sigmax2)
            solve = lambda state : sesolve(self.H, state, self.total_time_steps, e_ops=[sigmax1, sigmax2, sigmaz1, sigmaz2], progress_bar=None)
            all_states = [solve(state).expect for state in self.initial_states]
            states = [np.asarray(states, dtype='double') for states in all_states] 
            states = np.asarray([np.column_stack([state[0], state[1], state[2], state[3]]) for state in states])
            expect_data.append(states)
        
        expect_data = np.asarray(expect_data)
        self.total_expect_data = expect_data.reshape(self.num_trajs, time_steps, 4)
        self.train_time_steps = self.total_time_steps[np.where(self.total_time_steps <= stop)]
        self.train_expect_data = self.total_expect_data[:,:self.train_time_steps.shape[0],:]

if __name__ == '__main__':
    data = TwoQubitDataset()
    print(data.total_expect_data.shape[0])


