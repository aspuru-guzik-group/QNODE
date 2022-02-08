import numpy as np
from model import latent_ode, save_model
from dataloader import StochasticTwoLevelDataset
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--type', type=str, default='closed')
parser.add_argument('--obs_dim', type=int, default=3)
parser.add_argument('--latent_dim', type=int, default=6)
parser.add_argument('--rnn_nhidden', type=int, default=48)
parser.add_argument('--nhidden', type=int, default=48)
parser.add_argument('--epochs', type=int, default=7500)
parser.add_argument('--lr', type=float, default=0.004)
args = parser.parse_args()

def save_dataset(data, dataset_name):
    torch.save(data, './saved_datasets/{}.pt'.format(dataset_name))

def load_dataset(dataset_name):
    return torch.load('./saved_datasets/{}.pt'.format(dataset_name))

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # initializing the dataset
    data = StochasticTwoLevelDataset(dataset_type=args.type)
    save_dataset(data, '{}-{}'.format(args.type, args.seed))
    print('dataset {}-{} saved'.format(args.type, args.seed))

    #initializing the model
    trajs = torch.from_numpy(data.train_expect_data).float()
    ts = torch.from_numpy(data.train_time_steps).float()
    model = latent_ode(obs_dim=args.obs_dim, latent_dim=args.latent_dim, nhidden=args.nhidden, 
        rnn_nhidden=args.rnn_nhidden, lr=args.lr, batch=data.train_expect_data.shape[0])

    model.train(trajs, ts, args.epochs)
    save_model(model, 'trained_{}_{}_{}_{}-{}'.format(args.type, args.obs_dim, args.latent_dim, args.rnn_nhidden, args.nhidden))



