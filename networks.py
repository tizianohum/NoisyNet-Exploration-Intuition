from collections import OrderedDict

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        """
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(obs_dim, hidden_dim)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden_dim, hidden_dim)),
                    ("relu2", nn.ReLU()),
                    ("out", nn.Linear(hidden_dim, n_actions)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)


# Ablation of the QNetwork class to create a NoisyQNetwork
class NoisyQNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action with noisy layers.

    Architecture: the last two layers are replaced with NoisyLinear layers according to the NoisyNet paper of Fortunato et al.
    Input → Linear(obs_dim→hidden_dim) → ReLU
        → NoisyLinear(hidden_dim→hidden_dim) → ReLU
        → NoisyLinear(hidden_dim→n_actions)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        """
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("Linear1", nn.Linear(obs_dim, hidden_dim)),
                    ("relu1", nn.ReLU()),
                    ("Noisy1", NoisyLinear(hidden_dim, hidden_dim)),  # sigma_init is the initial value for the standard deviation of the noise
                    ("relu2", nn.ReLU()),
                    ("Noisy2", NoisyLinear(hidden_dim, n_actions)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)


class NoisyLinear(nn.Module):
    '''From https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/network.py'''
    '''edited slightly to fit into our project'''
    """
    Parameters
    ----------
    in_features : int
        Number of input features for this layer
    out_features : int
        Number of output features for this layer
    sigma_init : float
        Initial value for the standard deviation of the noise. 0.5 is the proposed value in the NoisyNet paper.
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features)) # these parameters (mu and sigma for weights and biases) will be optimized during training
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features)) # this is the actual noise, which is part of the model but will not be optimized or updated

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters() # for mu and sigma
        self.reset_noise() # for epsilon

    def forward(self, x):
        if self.training: #double check @MATTHIS !!! Das ist die einzige Zeile, bei der ich mir nicht sicher bin, ob und wie die funktioniert
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon) #calculation of the noisy weights and biases
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self): # this function is only called once at the beginning of training to initialize the weights and biases
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range) # according to the NoisyNet paper, mu is initialized uniformly in the range [-1/sqrt(in_features), 1/sqrt(in_features)]
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features)) # according to the NoisyNet paper, sigma is initialized as sigma_init / sqrt(in_features) with sigma_init = 0.5
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self): 
        # we use factorized Gaussian noise, which keeps computational costs low.
        # The noise is generated by scaling the standard deviation with a random variable that is sampled from a normal distribution.
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i)) # here we create the matrix of epsilon_i_j as a factorized outer product of two Gaussian noise vectors
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        # Generates scaled Gaussian noise.
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x