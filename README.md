# QNODE: discovering quantum dynamics using latent neural ODEs

### Prerequisites

| command | min. version |
|:-:|:-:|
| torchdiffeq  | 0.0.1 |
| numpy  | 1.17.4  |
| Pytorch  | 1.4.0 |
| QuTip | 4.6.2  |
| matplotlib  | 3.4.3  |
| scikit-learn  | 0.23.1  |
| imageio  | 2.6.1  |

## Training Models

run `python3 train.py`

To train a model with different hyperparameters:
| command | meaning |
|:-:|:-:|
| --seed  | the torch and numpy random seed  |
| --epochs  | numbers of iterations the model with train on |
| --type | the open or closed dataset  |
| --obs_dim  | input dimensions  |
| --rnn_nhidden  | rnn layer size  |
| --nhidden  | decoder layer size  |
| --latent_dim  | latent space size |
| --lr | learning rate  |

Example: 
`python3 train.py --seed 1 --epochs 5000 --lr 5e-3 --type closed`

## Generating Results

run `./create_plots.sh`

<sub><sup>Note: you might have to run `chmod +x create_plots.sh`</sup></sub>
