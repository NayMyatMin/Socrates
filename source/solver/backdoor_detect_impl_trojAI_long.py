import os
import time
import statistics
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from model.lib_models import *
from model.lib_layers import *
from wrn import WideResNet
from nn_utils import *
from solver.refinement_impl import Poly
from utils import *


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


class SubMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = x  # cross entropy in pytorch already includes softmax
        return output


class BackdoorDetectImpl:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_kwargs = {'batch_size': 100}
        self.test_kwargs = {'batch_size': 1000}
        self.transform = transforms.ToTensor()
        self.acc_th = 40.0
        self.ano_th = -1.5
        self.d0_th = 0.0
        self.d2_th = 0.0
        self.num_of_epochs = 100
        self.norm = 2
        self.size_input = None
        self.size_last = None

    def transfer_model(self, model, sub_model, dataset):
        params = model.named_parameters()
        sub_params = sub_model.named_parameters()
        dict_params = dict(sub_params)
        for name, param in params:
            if dataset == 'MNIST':
                if 'main.13.weight' in name:
                    dict_params['fc.weight'].data.copy_(param.data)
                elif 'main.13.bias' in name:
                    dict_params['fc.bias'].data.copy_(param.data)
            elif dataset == 'CIFAR-10':
                if 'fc.weight' in name:
                    dict_params['fc.weight'].data.copy_(param.data)
                elif 'fc.bias' in name:
                    dict_params['fc.bias'].data.copy_(param.data)
        sub_model.load_state_dict(dict_params)

    def get_clamp(self, epoch, batch, max_clamp, min_clamp, num_batches, num_epochs):
        t = epoch * num_batches + batch
        T = num_batches * num_epochs
        clamp = (max_clamp - min_clamp) * 0.5 * (1 + np.cos(np.pi * t / T)) + min_clamp
        return clamp

    def generate_trigger(self, model, dataloader, size, target, norm, minx, maxx=None):
        delta = torch.zeros(size, requires_grad=True, device=next(model.parameters()).device)
        eps = 0.001
        lamb = 1
        optimizer = torch.optim.Adam([delta], lr=0.01)

        def adjust_learning_rate(optimizer, epoch, batch, num_batches):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = 0.01 * (0.1 ** (epoch // 30))
            lr = lr * (1 - batch / num_batches)  # reduce learning rate linearly as we approach the end of the epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for epoch in range(self.num_of_epochs):
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(delta.device)
                y = y.to(delta.device)
                optimizer.zero_grad()
                clamp = self.get_clamp(epoch, batch, max_clamp=10, min_clamp=1, num_batches=len(dataloader),
                                    num_epochs=self.num_of_epochs)
                x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                target_tensor = torch.full_like(y, target, device=next(model.parameters()).device, dtype=torch.long)
                pred = model(x_adv)
                loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(delta, norm) # NOTE: try replace lamb * torch.norm(delta, norm) with torch.norm(delta, norm) ** 2 
                loss.backward()
                optimizer.step()
                delta.data = torch.clamp(delta.data, -clamp, clamp)
                adjust_learning_rate(optimizer, epoch, batch, len(dataloader))
        return delta.detach().cpu()

    def check(self, model, dataloader, delta, target):
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
        root_dir = "benchmarking-cifar"
        for subdir, dirs, files in os.walk(root_dir):
            start_time = time.time()
            dist0_lst, dist2_lst, acc_lst = [], [], []

            for file in files:
                if file == "model.pt":
                    model_path = os.path.join(subdir, file)
                    print("Model Path:", model_path)
                    info_path = os.path.join(subdir, "info.json")
                    with open(info_path) as f:
                        info = json.load(f)

                        if info["dataset"] == "MNIST":
                            dataset = info["dataset"]
                            print('dataset =', dataset)
                            last_layer = 'main[11]'  # nn.BatchNorm1d(128)
                            model = load_model(MNIST_Network, model_path)
                            model.main[11].register_forward_hook(get_activation(last_layer))
                            sub_model = SubMNISTNet()
                            train_dataset = datasets.MNIST('./data', train=True, download=False, transform=self.transform)
                            test_dataset = datasets.MNIST('./data', train=False, transform=self.transform)
                            self.size_input, self.size_last = (28, 28), 128

                        elif info["dataset"] == "CIFAR-10":
                            dataset = info["dataset"]
                            print('dataset =', dataset)
                            last_layer = 'avgpool'
                            model = load_model(WideResNet, model_path)
                            model.avgpool.register_forward_hook(get_activation(last_layer))
                            sub_model = SubWideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0)
                            train_dataset = datasets.CIFAR10('./data', train=True, download=False, transform=self.transform)
                            test_dataset = datasets.CIFAR10('./data', train=False, transform=self.transform)
                            self.size_input, self.size_last = (3, 32, 32), 128

                    self.transfer_model(model, sub_model, dataset)
                    train_dataloader = DataLoader(train_dataset, **self.train_kwargs)
                    test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test_kwargs)

                    last_layer_test_dataset = []
                    for batch, (x, y) in enumerate(test_dataloader):
                        model(x)
                        if dataset == 'MNIST':
                            last_layer_test_dataset.extend(F.relu(activation[last_layer]).detach().numpy())
                        elif dataset == 'CIFAR-10':
                            last_layer_test_dataset.extend(torch.flatten(activation[last_layer], 1).detach().numpy())
                    last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)),
                                                             torch.Tensor(np.array(test_dataset.targets)))
                    last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **self.test_kwargs)
                    dataloader = last_layer_test_dataloader

                    print_line = '\n###############################\n'
                    
                    for target in range(10):
                        delta = self.generate_trigger(sub_model, dataloader, self.size_last, target, self.norm, minx=0.0)
                        mask = abs(delta) < 0.1
                        delta[mask] = 0

                        print(print_line)
                        acc = self.check(sub_model, dataloader, delta, target)
                        dist0 = torch.norm(delta, 0)
                        dist2 = torch.norm(delta, 2)
                        print('target = {}, delta = {}, dist0 = {}, dist2 = {}'.format(target, delta[:10], dist0, dist2))
                        acc_lst.append(acc)
                        dist0_lst.append(dist0.detach().item())
                        dist2_lst.append(dist2.detach().item())

                    acc_lst = np.array(acc_lst)
                    dist0_lst = np.array(dist0_lst)
                    dist2_lst = np.array(dist2_lst)
                    med = np.median(dist0_lst)
                    dev_lst = abs(dist0_lst - med)
                    mad = np.median(dev_lst)
                    if mad > 0:
                        ano_lst = (dist0_lst - med) / mad
                    else:
                        ano_lst = dist0_lst - med

                    end_time = time.time()

                    print(print_line)
                    print('acc_lst = {}'.format(acc_lst))
                    print('dist0_lst = {}'.format(dist0_lst)) # NOTE: dist_lst here seems to be 0-norm distance already
                    print('dist2_lst = {}'.format(dist2_lst))
                    print('med = {}'.format(med))                    
                    print('dev_lst = {}'.format(dev_lst))                    
                    print('mad = {}'.format(mad))                    
                    print('ano_lst = {}'.format(ano_lst))                    
                    print("Elapsed time:", end_time - start_time, "seconds")
                    print(print_line)

                    for target in range(10):
                        if acc_lst[target] >= self.acc_th and (ano_lst[target] <= self.ano_th or dist0_lst[target] <= self.d0_th or dist0_lst[target] <= self.d2_th):
                            print('Detect backdoor at target = {}'.format(target))


# - Add two variables self.d0_th and self.d2_th (lines 54-55) and use them as the condition in line 219. 
# Currently, we do not know which values we should use so just use 0.0. 
# We will change after collecting some samples of d0 and d2.
# - Change dist_lst to dist0_lst and add dist2_lst. 
# Collect dist0 and dist2 in lines 187-188 and add them to according lists.
# - Line 107, we will also try torch.norm(delta, norm) ** 2 besides lamb * torch.norm(delta, norm).

# Some points I am not sure about:

# - Previously, we already print dist_lst, which is the list of 0-norm delta, 
# so I suppose you should have this data already?

# Anyway, please check the code, feel free to refactor, and try to collect some data.
