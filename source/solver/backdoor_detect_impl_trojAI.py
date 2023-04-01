import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import auc
from torch.cuda.amp import autocast
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
        self.fc = nn.Linear(nChannels[3], num_classes)

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
        return x

class BackdoorDetectImpl:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_kwargs = {'batch_size': 100, 'num_workers': 6, 'pin_memory': True}
        self.test_kwargs = {'batch_size': 1000, 'num_workers': 6, 'pin_memory': True}
        self.transform = transforms.ToTensor()
        self.size_input = None; self.size_last = None
        self.num_of_epochs = 100
        self.norm = 2
        
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

    def generate_trigger(self, model, dataloader, size, target, norm, minx=0.0, maxx=1.0):
        device = next(model.parameters()).device
        delta, eps, lamb = torch.rand(size, device=device) * 0.1, 0.001, 0.5
        best_delta, best_loss = None, float('inf')
        clamp_values, trigger_quality_sum, trigger_size_sum, trigger_distortion_sum = [], 0, 0, 0
        num_samples = 0
        for epoch in range(self.num_of_epochs):
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(device)
                delta.requires_grad = True
                with autocast():
                    x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                    target_tensor = torch.full(y.size(), target, device=device)

                    pred = model(x_adv)
                    trigger_success = (torch.argmax(pred, dim=1) == target).sum().item()
                    trigger_quality_sum += trigger_success
                    trigger = x_adv - x
                    trigger_size_sum += torch.norm(trigger.view(trigger.size(0), -1), norm, dim=1).sum().item()
                    trigger_distortion_sum += torch.norm(delta, 2).item() * x.size(0)
                    num_samples += x.size(0)

                loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(delta, norm)
                loss.backward()
                grad_data = delta.grad.data

                clamp = self.get_clamp(epoch, batch, max_clamp=5, min_clamp=1, num_batches=len(dataloader),
                        num_epochs=self.num_of_epochs)
                delta = torch.clamp(delta - eps * grad_data.sign(), -clamp, clamp).detach()
                clamp_values.append(clamp)

            if loss < best_loss:
                best_delta, best_loss = delta.detach(), loss

        return best_delta, trigger_size_sum / num_samples, trigger_distortion_sum / num_samples

    def check(self, model, dataloader, delta, target):
        model.eval() 
        device = next(model.parameters()).device
        correct = 0
        total = 0
        with torch.no_grad():  
            for x, y in dataloader:
                x = x.to(device)
                x_adv = torch.add(x, delta)
                target_tensor = torch.full(y.size(), target, device=device)
                pred = model(x_adv)
                correct += (pred.argmax(1) == target_tensor).type(torch.int).sum().item()
                total += y.size(0)
        return correct / total
    
    def load_info(self, model_path):
        try:
            with open(os.path.join(os.path.dirname(model_path), "info.json")) as f:
                info = json.load(f)
                return info
        except FileNotFoundError as e:
            print(f"Error loading info.json: {e}")
            raise

    def get_dataset(self, dataset_name):
        dataset_configs = {
            "MNIST": {
                "train_dataset": datasets.MNIST('./data', train=True, download=False, transform=self.transform),
                "test_dataset": datasets.MNIST('./data', train=False, transform=self.transform),
                "input_size": (28, 28),
                "last_size": 128
            },
            "CIFAR-10": {
                "train_dataset": datasets.CIFAR10('./data', train=True, download=False, transform=self.transform),
                "test_dataset": datasets.CIFAR10('./data', train=False, transform=self.transform),
                "input_size": (3, 32, 32),
                "last_size": 128
            }
        }

        return dataset_configs.get(dataset_name)

    def load_and_prepare_model(self, model_path):
        info = self.load_info(model_path)
        dataset = info["dataset"]
        dataset_config = self.get_dataset(dataset)
        print(f'Dataset = {dataset}\n')
        if dataset == "MNIST":
            last_layer = 'main[11]'
            model = load_model(MNIST_Network, model_path)
            model.main[11].register_forward_hook(get_activation(last_layer))
            sub_model = SubMNISTNet()

        elif dataset == "CIFAR-10":
            last_layer = 'avgpool'
            model = load_model(WideResNet, model_path)
            model.avgpool.register_forward_hook(get_activation(last_layer))
            sub_model = SubWideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0)

        model.to(self.device)
        sub_model.to(self.device)

        self.transfer_model(model, sub_model, dataset)

        self.size_input = dataset_config["input_size"]
        self.size_last = dataset_config["last_size"]

        return model, sub_model, dataset, dataset_config["train_dataset"], dataset_config["test_dataset"], last_layer

    def get_last_layer_activations(self, model, test_dataset, last_layer, dataset):
        model.eval()  
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test_kwargs)
        last_layer_test_dataset = []
        with torch.no_grad():  
            for batch, (x, y) in enumerate(test_dataloader):
                x = x.to(self.device)
                model(x)
                if dataset == 'MNIST':
                    last_layer_test_dataset.extend(F.relu(activation[last_layer]).cpu().detach().numpy())
                elif dataset == 'CIFAR-10':
                    last_layer_test_dataset.extend(torch.flatten(activation[last_layer], 1).cpu().detach().numpy())
        last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)),
                                                torch.Tensor(np.array(test_dataset.targets)))
        last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **self.test_kwargs)
        return last_layer_test_dataloader

    def evaluate_triggers(self, model, sub_model, dataloader):
        acc_lst, dist0_lst, dist2_lst = [], [], []

        for target in range(10):
            delta, trigger_size, trigger_distortion = self.generate_trigger(sub_model, dataloader, self.size_last, target, self.norm, minx=0.0, maxx=None)
            delta = torch.where(abs(delta) < 0.1, delta - delta, delta)
            acc = self.check(sub_model, dataloader, delta, target)
            dist0 = torch.norm(delta, 0)
            dist2 = torch.norm(delta, 2)
            acc_lst.append(acc)
            dist0_lst.append(dist0.detach().item())
            dist2_lst.append(dist2.detach().item())
            print('trigger_size = {}, trigger_distortion = {}'.format(trigger_size, trigger_distortion))
            print('target = {}, delta = {}, dist0 = {}, dist2 = {}\n'.format(target, delta[:10], dist0, dist2))
        return acc_lst, dist0_lst, dist2_lst

    def detect_backdoors(self, acc_lst, dist0_lst, dist2_lst):
        detected_backdoors = []
        med = np.median(dist0_lst)
        dev_lst = abs(np.array(dist0_lst) - med)
        mad = np.median(dev_lst)
        epsilon = 1e-9
        ano_lst = (np.array(dist0_lst) - med) / (mad + epsilon)

        print('Accuracy list for all targets: {}'.format(acc_lst))
        print('Distance list (dist0_lst): {}'.format(dist0_lst))
        print('Distance list (dist2_lst): {}'.format(dist2_lst))
        print('Median of the distance list (dist0_lst): {}'.format(med))
        print('Absolute deviations from the median: {}'.format(dev_lst))
        print('Median absolute deviation (MAD): {}'.format(mad))
        print('Anomaly scores: {}'.format(ano_lst))  
        
        # Calculate the adaptive thresholds using the percentile-based approach
        acc_th = np.percentile(acc_lst, 95)
        mad = np.median(np.abs(dist0_lst - np.median(dist0_lst)))
        if mad != 0.0:
            ano_lst = np.abs(dist0_lst - np.median(dist0_lst)) / mad
            ano_th = np.percentile(ano_lst, 3)
        else:
            ano_th = np.inf

        d0_th = np.percentile(dist0_lst, 3)
        d2_th = np.percentile(dist2_lst, 3)
        print('acc_th = {}, ano_th = {}, d0_th = {}, d2_th = {}\n'.format(acc_th, ano_th, d0_th, d2_th))
        for target in range(10):
            if acc_lst[target] >= acc_th and (ano_th == np.inf or ano_lst[target] <= ano_th or dist0_lst[target] <= d0_th or dist2_lst[target] <= d2_th):
                detected_backdoors.append(target)
        return detected_backdoors
    
    def calculate_tpr_fpr(self, thresholds, TP, TN, FP, FN):
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            tpr = TP / (TP + FN)
            fpr = FP / (FP + TN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return tpr_list, fpr_list

    def solve(self, model, assertion, display=None):
        clean_root_dir = "25C-Benign"
        backdoor_root_dir = "25C-Backdoor"
        total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0
        
        for clean_subdir, _, files in os.walk(clean_root_dir):
            start_time = time.time()
            for file in files:
                if file == "model.pt":
                    print('Clean')
                    model_path = os.path.join(clean_subdir, file)
                    model, sub_model, dataset, train_dataset, test_dataset, last_layer = self.load_and_prepare_model(model_path)
                    dataloader = self.get_last_layer_activations(model, test_dataset, last_layer, dataset)
                    acc_lst, dist0_lst, dist2_lst = self.evaluate_triggers(model, sub_model, dataloader)
                    detected_backdoors = self.detect_backdoors(acc_lst, dist0_lst, dist2_lst)

                    if detected_backdoors:
                        print("(Wrong) Detected backdoors at targets:", detected_backdoors)
                        total_FP += 1
                    else:
                        print("No backdoors detected.")
                        total_TN += 1
                    end_time = time.time()        
                    print("Elapsed time:", end_time - start_time, "seconds")
                    print(f"\n{'*' * 100}\n")

        for backdoor_subdir, _, files in os.walk(backdoor_root_dir):
            start_time = time.time()
            for file in files:
                if file == "model.pt":
                    print('Backdoor')
                    model_path = os.path.join(backdoor_subdir, file)
                    model, sub_model, dataset, train_dataset, test_dataset, last_layer = self.load_and_prepare_model(model_path)
                    dataloader = self.get_last_layer_activations(model, test_dataset, last_layer, dataset)
                    acc_lst, dist0_lst, dist2_lst = self.evaluate_triggers(model, sub_model, dataloader)
                    detected_backdoors = self.detect_backdoors(acc_lst, dist0_lst, dist2_lst)

                    if detected_backdoors:
                        print("Detected backdoors at targets:", detected_backdoors)
                        total_TP += 1
                    else:
                        print("(Wrong) No backdoors detected.")
                        total_FN += 1
                    end_time = time.time()    
                    print("Elapsed time:", end_time - start_time, "seconds")
                    print(f"\n{'*' * 100}\n")

        thresholds = [0.5] 
        tpr_list, fpr_list = self.calculate_tpr_fpr(thresholds, total_TP, total_TN, total_FP, total_FN)
        print(f'tpr_list: {tpr_list}, fpr_list: {fpr_list}')

        end_time = time.time()
        precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
        recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        print(f"total_TP: {total_TP:.2f}, total_TN: {total_TN:.2f}, total_FP: {total_FN:.2f}, total_FN: {total_FN:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}, Accuracy: {accuracy:.2f}")