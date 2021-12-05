import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ])
        # need a mu/sigma for each action, hence output dimensions=3
        self.mu = nn.Linear(32, 3)
        self.sigma = nn.Linear(32, 3)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # TODO: clip mu/sigma if greater than what is allowed by sim
        x = self.layers(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = self.softplus(sigma) + 1e-5

        norm_dist = torch.distributions.normal.Normal(mu, sigma)
        action = norm_dist.sample()
        print(action)
        action = torch.clamp(action, min=torch.Tensor([-15, 0, -15]), max=torch.Tensor([15, 100, 15]))
        return action, norm_dist


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
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

