import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

def save_model(model, model_name):
    torch.save(model.func.state_dict(), './saved_models/{}_func.pt'.format(model_name))
    torch.save(model.dec.state_dict(), './saved_models/{}_dec.pt'.format(model_name))
    torch.save(model.rec.state_dict(), 'saved_models/{}_rec.pt'.format(model_name))
    torch.save(model.epsilon, './saved_models/{}_epsilon.pt'.format(model_name))

def load_model(model, model_name):
    model.func.load_state_dict(torch.load('./saved_models/{}_func.pt'.format(model_name)))
    model.dec.load_state_dict(torch.load('./saved_models/{}_dec.pt'.format(model_name)))
    model.rec.load_state_dict(torch.load('./saved_models/{}_rec.pt'.format(model_name)))
    try:
        model.epsilon = torch.load('./saved_models/{}_epsilon.pt'.format(model_name))
    except:
        pass
    model.func.eval()
    model.dec.eval()
    model.rec.eval()

def load(type):
    if type == 'open':
        data = torch.load('saved_datasets/open_6_53_53_0.007.pt')
        model = latent_ode(batch=1080, obs_dim=3, latent_dim=6, nhidden=53, rnn_nhidden=53, lr=0.007, beta=1, extra_decode=True)
        load_model(model, 'open_6_53_53_0.007')
    elif type == 'closed':
        data = torch.load('saved_datasets/closed_6_48_48_0.004.pt') 
        model = latent_ode(batch=1080, obs_dim=3, latent_dim=6, nhidden=48, rnn_nhidden=48, lr=0.004, extra_decode=True)
        load_model(model, 'closed_6_48_48_0.004')
    elif type == 'two':
        data = torch.load('saved_datasets/two_8_170_170_0.002.pt') 
        model = latent_ode(batch=1080, obs_dim=4, latent_dim=8, nhidden=170, rnn_nhidden=170, lr=0.002, extra_decode=True)
        load_model(model, 'two_8_170_170_0.002')

    return data, model

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        out = self.elu(out)
        out = self.fc4(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=3, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, batch=0):
        if batch == 0:
            return torch.zeros(self.nbatch, self.nhidden)
        else:
            return torch.zeros(batch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20, extra=False):
        super(Decoder, self).__init__()
        self.extra = extra
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.tanh(out)
        if self.extra:
            out = self.fc2(out)
            out = self.tanh(out)
            out = self.fc3(out)
        else:
            out = self.fc3(out)
        return out

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class latent_ode:
    def __init__(self, obs_dim=2, latent_dim=4, nhidden=20, rnn_nhidden=25, lr=1e-2, batch=1000, beta=1, extra_decode=False):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.rnn_nhidden = rnn_nhidden
        self.beta = beta
        self.epsilon = None

        self.func = LatentODEfunc(latent_dim, nhidden)
        self.rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch)
        self.dec = Decoder(latent_dim, obs_dim, nhidden, extra=extra_decode)
        self.params = (list(self.func.parameters()) + list(self.dec.parameters()) + list(self.rec.parameters()))
        self.lr = lr
        self.optimizer = optim.Adam(self.params, lr=self.lr)

    def train(self, trajs, ts, num_epochs):
        # dataset parameters
        num_ts = ts.size(0)
        beta = self.beta
        for itr in range(1, num_epochs + 1):
            self.optimizer.zero_grad()
            h = self.rec.initHidden()
            for t in reversed(range(num_ts)):
                obs = trajs[:, t, :]
                out, h = self.rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            epsilon = torch.randn(qz0_mean.size())
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(self.func, z0, ts).permute(1, 0, 2)
            pred_x = self.dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()) + 0.3
            noise_logvar = 2. * torch.log(noise_std_)
            logpx = log_normal_pdf(
                trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size())
            analytic_kl = beta * normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            self.optimizer.step()

            if itr == num_epochs:
                self.epsilon = epsilon

            print('Epoch: {}, elbo: {:.4f}'.format(itr, loss))

    def encode(self, trajs, ts, reconstruct=True):
        if (reconstruct):
            with torch.no_grad():
                num_ts = ts.size(0)
                # sample from trajectorys' approx. posterior
                h = self.rec.initHidden(batch=trajs.shape[0])
                for t in reversed(range(num_ts)):
                    obs = trajs[:, t, :]
                    out, h = self.rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
                z0 = qz0_mean #self.epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        else:
            with torch.no_grad():
                num_ts = ts.size(0)
                # sample from trajectorys' approx. posterior
                h = self.rec.initHidden(batch=trajs.shape[0])
                for t in reversed(range(num_ts)):
                    obs = trajs[:, t, :]
                    out, h = self.rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
                epsilon = torch.randn(qz0_mean.size())
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        return z0

    def decode(self, z0, ts):
        with torch.no_grad():
            if len(z0.shape) == 1:
                pred_z = odeint(self.func, z0, ts)
            else:
                pred_z = odeint(self.func, z0, ts).permute(1, 0, 2)
            pred_x = self.dec(pred_z)
        return pred_x

    def latent_dynamics(self, trajs, enc_ts, dec_ts, recontruct=True):
        z0 = self.encode(trajs, enc_ts, recontruct)
        print('z0',z0.shape)
        with torch.no_grad():
            if len(z0.shape) == 1:
                pred_z = odeint(self.func, z0, dec_ts)
            else:
                pred_z = odeint(self.func, z0, dec_ts).permute(1, 0, 2)
        return pred_z


    def MSE(self, trajs, train_ts):
        z0 = self.encode(trajs, train_ts)
        pred_x = self.decode(z0, train_ts)

        mse_errors = np.mean((trajs.numpy() - pred_x.numpy()) ** 2, axis=1)
        mse_errors = np.mean(mse_errors, axis=1)
        avg_mse = np.mean(mse_errors)

        return avg_mse, mse_errors
    
