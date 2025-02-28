from torch import nn
import torch
import random

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class LR(nn.Module):
    def __init__(self, args):
        super(LR, self).__init__()
        if args.dataset == 'synthetic':
            self.linear = nn.Linear(60, args.num_classes)
        elif args.dataset == 'femnist':
            self.linear = nn.Linear(784, args.num_classes)
        elif args.dataset == 'synthetic_fairness':
            self.linear = nn.Linear(2, args.num_classes)
        elif args.dataset == 'adult':
            self.linear = nn.Linear(17, args.num_classes)
        elif args.dataset == 'german':
            self.linear = nn.Linear(10, args.num_classes)
        elif args.dataset == 'compas':
            self.linear = nn.Linear(9, args.num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x


class CNNFemnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        return self.out(x)