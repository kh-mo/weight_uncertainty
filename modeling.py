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
    def __init__(self, mu, rho, args):
        super(Gaussian, self).__init__()
        self.args = args
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.normal.Normal(0, 1)
        # self.sigma = torch.log(1+torch.exp(self.rho)).to(device)

    @property
    def sigma(self):
        return torch.log(1+torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.args.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2*math.pi))
                -torch.log(self.sigma)
                -((input-self.mu)**2) / (2 * (self.sigma**2))).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2, args):
        super(ScaleMixtureGaussian, self).__init__()
        self.args = args
        self.pi = pi
        self.sigma1 = torch.FloatTensor([math.exp(sigma1)]).to(self.args.device)
        self.sigma2 = torch.FloatTensor([math.exp(sigma2)]).to(self.args.device)
        self.gaussian1 = torch.distributions.normal.Normal(0, self.sigma1)
        self.gaussian2 = torch.distributions.normal.Normal(0, self.sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super(BayesianLinear, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, self.args)
        # bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(self.out_dim).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, self.args)
        # prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.args.pi, self.args.sigma_1, self.args.sigma_2, self.args)
        self.bias_prior = ScaleMixtureGaussian(self.args.pi, self.args.sigma_1, self.args.sigma_2, self.args)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample):
        if self.training or sample:
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
    def __init__(self, args):
        super(BayesianNetwork, self).__init__()
        self.args = args
        self.linear1 = BayesianLinear(784, 400, self.args)
        self.linear2 = BayesianLinear(400, 400, self.args)
        self.linear3 = BayesianLinear(400, 10, self.args)

    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
        x = f.relu(self.linear1(x, sample))
        x = f.relu(self.linear2(x, sample))
        x = f.log_softmax(self.linear3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.linear1.log_prior + self.linear2.log_prior + self.linear3.log_prior

    def log_variational_posterior(self):
        return self.linear1.log_variational_posterior + self.linear2.log_variational_posterior + self.linear3.log_variational_posterior

    def sample_elbo(self, data, target):
        outputs = torch.zeros(self.args.samples, data.shape[0], self.args.classes).to(self.args.device)
        log_priors = torch.zeros(self.args.samples).to(self.args.device)
        log_variational_posteriors = torch.zeros(self.args.samples).to(self.args.device)
        for i in range(self.args.samples):
            outputs[i] = self.forward(data)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = f.nll_loss(outputs.mean(0), target, size_average=False).to(self.args.device)
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss#, log_prior, log_variational_posterior, negative_log_likelihood

def train(model, optimizer, epoch, args):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        model.zero_grad()
        loss = model.sample_elbo(data, target)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    print("epoch : {}, loss : {}".format(epoch, sum(loss_list)/len(loss_list)))
    if epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_model/weight_uncertainty_model" +
                                                        "_epoch" + str(epoch+1) +
                                                        "_batch_size" + str(args.batch_size) +
                                                        "_lr" + str(args.lr) +
                                                        "_sigma1" + str(args.sigma_1) +
                                                        "_sigma2" + str(args.sigma_2) +
                                                        "_pi" + str(args.pi)))

if __name__ == "__main__":
    # hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sigma_1", type=int, default=0)
    parser.add_argument("--sigma_2", type=int, default=-6)
    parser.add_argument("--pi", type=int, default=0.5)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=int, default=0.0001)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # folder for download
    try:
        os.makedirs("raw_data")
    except FileExistsError as e:
        pass

    # download data
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist_data = dset.MNIST('./raw_data/', train=True, download=True, transform=transform)
    train_loader = data.DataLoader(train_mnist_data, batch_size=args.batch_size)
    NUM_BATCHES = len(train_loader)

    # modeling
    model = BayesianNetwork(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, args)

    # save model
    try:
        os.makedirs("saved_model")
    except FileExistsError as e:
        pass
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "saved_model/weight_uncertainty_model" +
                                                "_epoch" + str(args.epochs) +
                                                "_batch_size" + str(args.batch_size) +
                                                "_lr" + str(args.lr) +
                                                "_sigma1" + str(args.sigma_1) +
                                                "_sigma2" + str(args.sigma_2) +
                                                "_pi" + str(args.pi)))
