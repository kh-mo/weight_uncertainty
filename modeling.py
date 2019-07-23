import os

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

class Gaussian:
    def __init__(self, mu, sig):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.sig = sig
        self.normal = torch.distributions.Normal(0,1)

    def sigma(self):
        return torch.log1p(torch.exp(self.sig))

    def sample(self):
        epsilon = self.normal.sample(self.sig.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        a1 = -math.log(math.sqrt(2 * math.pi))
        a2 = -torch.log(self.sigma)
        a3 = -((input-self.mu)**2) / (2 * self.sigma ** 2)
        return (a1 + a2 + a3).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(BayesianLinear, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        # weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(self.out_feature, self.in_feature).uniform_(-0.2, 0.2))
        self.weight_sig = nn.Parameter(torch.Tensor(self.out_feature, self.in_feature).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_sig)

        # bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(self.out_features).uniform_(-0.2, 0.2))
        self.bias_sig = nn.Parameter(torch.Tensor(self.out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_sig)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super(BayesianNetwork, self).__init__()
        self.l1 = nn.Linear(784, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x

if __name__ == "__main__":
    # folder for download
    try:
        os.makedirs("./raw_data")
    except FileExistsError as e:
        pass

    # load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_mnist_data = dset.MNIST('./raw_data/', train=True, download=True, transform=transform)
    test_mnist_data = dset.MNIST('./raw_data/', train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_mnist_data)
    test_loader = data.DataLoader(test_mnist_data)

    # modeling
    model = BayesianNetwork()

    # training
    for sample, label in train_loader:
        output = model(sample)
        break
