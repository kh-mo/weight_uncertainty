import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as f
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Gaussian(object):
    def __init__(self, mu, rho):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.normal.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log(1+torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2*math.pi))
                -torch.log(self.sigma)
                -((input-self.mu)**2) / (2 * (self.sigma**2))).sum()

class ScaleMixtureGaussian():
    def __init__(self, pi, sigma1, sigma2):
        super(ScaleMixtureGaussian, self).__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.normal.Normal(0, self.sigma1)
        self.gaussian2 = torch.distributions.normal.Normal(0, self.sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BayesianLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.1, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.1, 0.1))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.1, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.1, 0.1))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input):
        if self.training:
            weight = self.weight.sample()
            bias = self.bias.sample()
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias) # log P(w)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias) # log q(w|theta)
        else:
            weight = self.weight.mu
            bias = self.bias.mu
            self.log_prior = 0
            self.log_variational_posterior = 0
        return f.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.linear1 = BayesianLinear(784, 400)
        self.linear2 = BayesianLinear(400, 400)
        self.linear3 = BayesianLinear(400, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = f.relu(self.l1(x))
        x = f.relu(self.l2(x))
        x = f.log_softmax(self.l3(x), dim=1)
        return x

    def sample_elbo(self):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(device)

        return loss

def train(model, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        loss = model.sample_elbo(data, target)
        loss.backward()
        optimizer.step()

    return



if __name__ == "__main__":
    # hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PI = 0.5
    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
    CLASSES = 10
    BATCH_SIZE = 100

    # folder for download
    try:
        os.makedirs("raw_data")
    except FileExistsError as e:
        pass

    # download data
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist_data = dset.MNIST('./raw_data/', train=True, download=True, transform=transform)
    test_mnist_data = dset.MNIST('./raw_data/', train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_mnist_data)
    test_loader = data.DataLoader(test_mnist_data)

    # modeling
    model = BayesianNetwork().to(device)
    optimizer = optim.Adam(model.parameters())

    # training
    for epoch in range(args.epoch):
        train(model, optimizer, epoch)

    # folder for save model
    try:
        os.makedirs("saved_model")
    except FileExistsError as e:
        pass
