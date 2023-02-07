import torch
import random
import numpy as np

from model.lib_models import *
from model.lib_layers import *

from poly_utils import *
from solver.refinement_impl import Poly

from utils import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import gurobipy as gp
from gurobipy import GRB

import math
import ast

from nn_utils import *

from wrn import WideResNet
import time
import statistics

###################################################### Troj-AI WRN CIFAR-10
class SubWideResNet(nn.Module):
    def __init__(self, depth=40, num_classes=10, widen_factor=2, dropRate=0.0):
        super(SubWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
    def forward(self, x):
        out = torch.flatten(x, 1)
        return self.fc(out)

###################################################### Troj-AI MNIST
class SubMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = x # cross entropy in pytorch already includes softmax
        return output
##################################################

class BackdoorDetectImpl():
    def __transfer_model(self, model, sub_model, dataset):
        params = model.named_parameters()
        sub_params = sub_model.named_parameters()

        dict_params = dict(sub_params)

        for name, param in params:
            if dataset == 'mnist' and ('main[13].weight' in name or 'main[13].bias' in name):
                dict_params[name].data.copy_(param.data)
            elif dataset == 'CIFAR-10' and ('model.fc.weight' in name or 'model.fc.bias' in name):
                dict_params[name].data.copy_(param.data)
        
        sub_model.load_state_dict(dict_params)


    def __generate_trigger(self, model, dataloader, num_of_epochs, size, target, norm, minx = None, maxx = None):
        delta, eps, lamb = torch.zeros(size), 0.001, 1

        for epoch in range(num_of_epochs):
            for batch, (x, y) in enumerate(dataloader):
                delta.requires_grad = True
                x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                target_tensor = torch.full(y.size(), target)

                pred = model(x_adv)
                loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(delta, norm)

                loss.backward()
                
                grad_data = delta.grad.data
                delta = torch.clamp(delta - eps * grad_data.sign(), -10.0, 10.0).detach()

        return delta


    def __check(self, model, dataloader, delta, target):
        size = len(dataloader.dataset)
        correct = 0

        for batch, (x, y) in enumerate(dataloader):
            x_adv = torch.clamp(torch.add(x, delta), 0.0)
            target_tensor = torch.full(y.size(), target)

            pred = model(x_adv)

            correct += (pred.argmax(1) == target_tensor).type(torch.int).sum().item()

        correct = correct / size * 100
        print('target = {}, test accuracy = {}'.format(target, correct))

        return correct


    def solve(self, model, assertion, display=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_kwargs, test_kwargs = {'batch_size': 100}, {'batch_size': 1000}
        transform = transforms.ToTensor()

        acc_th, ano_th = 40.0, -1.5

        # dataset = 'CIFAR-10'
        dataset = 'mnist'
        num_of_epochs = 100
        dist_lst, acc_lst = [], []
        norm = 2

        print('dataset =', dataset)

        ############################################################# - MNIST

        if dataset == 'mnist':
            file_name = './backdoor_models/model.pt'
            last_layer = 'main[11]' # nn.BatchNorm1d(128)

            model = load_model(MNIST_Network, file_name)
            print(model.eval())
            model.main[11].register_forward_hook(get_activation(last_layer))
        
            sub_model = SubMNISTNet()
            
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)

            size_input, size_last = (28, 28), 128
        ############################################################# - CIFAR-10

        elif dataset == 'CIFAR-10':
            file_name = './backdoor_models/model-c.pt'
            last_layer = 'avgpool'
            
            model = load_model(WideResNet, file_name)
            print(model.eval())
            model.avgpool.register_forward_hook(get_activation(last_layer))
        
            sub_model = SubWideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0)

            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

            size_input, size_last = (3, 32, 32), 128
        
        #############################################################

        self.__transfer_model(model, sub_model, dataset)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        last_layer_test_dataset = []
        for batch, (x, y) in enumerate(test_dataloader):
            model(x)
            if dataset == 'mnist':
                last_layer_test_dataset.extend(F.relu(activation[last_layer]).detach().numpy())
            elif dataset == 'CIFAR-10':
                last_layer_test_dataset.extend(torch.flatten(activation[last_layer], 1).detach().numpy())

        last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)), torch.Tensor(np.array(test_dataset.targets))) # create dataset
        last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **test_kwargs) # create dataloader

        # dataloader = test_dataloader
        dataloader = last_layer_test_dataloader

        for target in range(10):
            # delta = self.__generate_trigger(model, dataloader, num_of_epochs, size_input, target, norm, 0.0, 1.0)
            delta = self.__generate_trigger(sub_model, dataloader, num_of_epochs, size_last, target, norm, 0.0)
            delta = torch.where(abs(delta) < 0.1, delta - delta, delta)

            print('\n###############################\n')

            # acc = self.__check(model, dataloader, delta, target)
            acc = self.__check(sub_model, dataloader, delta, target)
            dist = torch.norm(delta, 0)
            print('\ntarget = {}, delta = {}, dist = {}\n'.format(target, delta[:10], dist))

            acc_lst.append(acc)
            dist_lst.append(dist.detach().item())

        print('\n###############################\n')

        acc_lst = np.array(acc_lst)
        print('acc_lst = {}'.format(acc_lst))

        dist_lst = np.array(dist_lst)
        print('dist_lst = {}'.format(dist_lst))

        med = statistics.median(dist_lst)
        print('med = {}'.format(med))
        
        dev_lst = abs(dist_lst - med)
        print('dev_lst = {}'.format(dev_lst))
        
        mad = statistics.median(dev_lst)
        print('mad = {}'.format(mad))
        
        ano_lst = (dist_lst - med) / mad
        print('ano_lst = {}'.format(ano_lst))

        print('\n###############################\n')

        for target in range(10):
            if acc_lst[target] >= acc_th and ano_lst[target] <= ano_th:
                print('Detect backdoor at target = {}'.format(target))