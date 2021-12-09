import torch
import numpy as np
from torch import nn


class Actor(nn.Module):
    ''' Actor model, which learns to choose actions that maximize its advantage over the current state '''
    def __init__(self):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(8, 32),
            nn.Sigmoid(),       # Choose sigmoid, so that outputs are between 0 and 1
            nn.Linear(32, 32),
            nn.Sigmoid(),
        ])
        # need a mu/sigma for each action (thrust, gimble_angle, and fin_angle), hence output dimensions=3
        self.mu = nn.Linear(32, 3)
        self.sigma = nn.Linear(32, 3)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = self.softplus(sigma) + 0.0001

        # Generate a normal distribution, and sample the action from that.
        # This has the effect of exploring more initially, but focusing in on the mean as the actor learns more.
        norm_dist = torch.distributions.normal.Normal(mu, sigma)
        action = norm_dist.sample()

        # this step moves the actions into their appropriate spaces for each action.  0-1 for trust, +/- pi/18 for gimple angle, and +/- pi/4 for fin angle
        action = self.sigmoid(action) * torch.Tensor([1, np.pi/16, np.pi/2]) - torch.Tensor([0, np.pi/32,np.pi/4])
        return action, norm_dist


class Critic(nn.Module):
    ''' Create a model that learns to estimate values based on states '''
    def __init__(self):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(8, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        ])

    def forward(self, x):
        x = self.layers(x)
        return x



if __name__ == "__main__":
    a = Actor()
    c = Critic()
    print(a)
    print(c)

    x = torch.randint(0, 10, (6,), dtype=torch.float32)
    print(a(x))

