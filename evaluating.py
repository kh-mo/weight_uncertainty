from modeling import BayesianNetwork

import os
import argparse
import numpy as np

import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

def test_ensemble(model, args):
    model.eval()
    correct = 0
    corrects = np.zeros(args.test_sample+1, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            outputs = torch.zeros(args.test_sample+1, data.shape[0], args.classes).to(args.device)
            for i in range(args.test_sample):
                outputs[i] = model(data, sample=True)
            outputs[args.test_sample] = model(data, sample=False)
            output = outputs.mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1] # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    for index, num in enumerate(corrects):
        if index < args.test_sample:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))

if __name__ == "__main__":
    # hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sigma_1", type=int, default=0)
    parser.add_argument("--sigma_2", type=int, default=-6)
    parser.add_argument("--pi", type=int, default=0.5)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--test_sample", type=int, default=10)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=int, default=0.0001)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download data
    transform = transforms.Compose([transforms.ToTensor()])
    test_mnist_data = dset.MNIST('./raw_data/', train=False, download=True, transform=transform)
    test_loader = data.DataLoader(test_mnist_data, batch_size=args.batch_size)
    TEST_SIZE = len(test_loader.dataset)

    model = BayesianNetwork(args).to(args.device)
    # model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/w_u")))

    model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_model/weight_uncertainty_model" +
                                                # "_epoch" + str(args.epoch) +
                                                "_epoch" + str(20) +
                                                "_batch_size" + str(args.batch_size) +
                                                "_lr" + str(args.lr) +
                                                "_sigma1" + str(args.sigma_1) +
                                                "_sigma2" + str(args.sigma_2) +
                                                "_pi" + str(args.pi))))

    test_ensemble(model, args)
