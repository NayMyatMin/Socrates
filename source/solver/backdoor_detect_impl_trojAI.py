import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from model.lib_models import *
from model.lib_layers import *
from wrn import WideResNet
from nn_utils import *
from utils import *

import torch.optim as optim
from torch.cuda.amp import GradScaler

alphabet = 'M'
acc_percent, ano_percent, d0_percent, d2_percent = 55, 1, 1, 1; lamb, lr = 0.1, 0.09
# clean_root_dir, backdoor_root_dir = f"dataset/benchmark-{alphabet}b", f"dataset/benchmark-{alphabet}"
clean_root_dir = f"dataset/10{alphabet}-Benign"; backdoor_root_dir = f"dataset/10{alphabet}-Backdoor"

class SubWideResNet(nn.Module):
    def __init__(self, depth=40, num_classes=10, widen_factor=2, dropRate=0.0):
        super(SubWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.fc = nn.Linear(nChannels[3], num_classes)

    def forward(self, x):
        out = torch.flatten(x, 1)
        return self.fc(out)

class SubMNIST(nn.Module):
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
        self.train_kwargs = {'batch_size': 100, 'num_workers': 8, 'pin_memory': True, 'drop_last':False}
        self.test_kwargs = {'batch_size': 128, 'num_workers': 8, 'pin_memory': True, 'drop_last':True}
        self.mnist_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
        self.cifar10_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
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

    def generate_trigger(self, model, dataloader, size, target, norm, exp_vect, minx=0.0, maxx=1.0, patience=5, min_improvement=1e-4):
        device = next(model.parameters()).device
        delta = torch.rand(size, device=device) * 0.1
        best_delta, best_loss = None, float('inf')
        patience_counter = 0
        
        optimizer = optim.Adam([delta], lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=False)
        scaler = GradScaler()

        for epoch in range(self.num_of_epochs):
            model.train()
            for batch, (x, y) in enumerate(dataloader):
                if exp_vect is None:
                    # generate trigger in form of a concrete vector at the last layer
                    # we do it by setting x to zero so delta will be the vector that we want
                    x = torch.zeros(x.size())

                # generate trigger in form of modification at the input layer
                # use normal x
                x = x.to(device)
                delta.requires_grad = True
                with torch.cuda.amp.autocast():
                    x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                    target_tensor = torch.full(y.size(), target, device=device)

                    pred = model(x_adv)
                    # print(f'pred_shape = {pred.shape}')
                    # print(f'pred = {pred}')
                    # print(f'target_tensor_shape = {target_tensor.shape}')
                    # print(f'target_tensor = {target_tensor}')

                    if exp_vect is None:
                        # compare with target tensor
                        loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(delta, norm)
                    else:
                        # compare with expected vector
                        # exp_vect = exp_vect.to(torch.long)
                        # print(f'exp_vet_shape = {exp_vect.shape}')
                        # print(f'exp_vet = {exp_vect}')
                        exp_vect_expanded = exp_vect.view(-1, 1).expand(-1, pred.shape[1])
                        loss = F.mse_loss(pred, exp_vect_expanded) + lamb * torch.norm(delta, norm)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                clamp = self.get_clamp(epoch, batch, max_clamp=5, min_clamp=1, num_batches=len(dataloader), num_epochs=self.num_of_epochs)
                delta.data = torch.clamp(delta.data, -clamp, clamp)

            model.eval()
            trigger_quality_sum, trigger_size_sum, trigger_distortion_sum, num_samples = 0, 0, 0, 0
            with torch.no_grad():
                for batch, (x, y) in enumerate(dataloader):
                    if exp_vect is None:
                        # generate trigger in form of a concrete vector at the last layer
                        # we do it by setting x to zero so delta will be the vector that we want
                        x = torch.zeros(x.size())

                    # generate trigger in form of modification at the input layer
                    # use normal x
                    x = x.to(device)
                    x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                    target_tensor = torch.full(y.size(), target, device=device)

                    pred = model(x_adv)
                    trigger_success = (torch.argmax(pred, dim=1) == target).sum().item()
                    trigger_quality_sum += trigger_success
                    trigger = x_adv - x
                    trigger_size_sum += torch.norm(trigger.view(trigger.size(0), -1), norm, dim=1).sum().item()
                    trigger_distortion_sum += torch.norm(delta, 2).item() * x.size(0)
                    num_samples += x.size(0)

                current_loss = trigger_quality_sum / num_samples
                scheduler.step(current_loss)
            
            # print(f'Epoch: {epoch}, Loss: {current_loss}')
            
            if best_loss - current_loss > min_improvement:
                best_loss = current_loss
                best_delta = delta.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # print(f'Early stopping at epoch {epoch}')
                break

        return best_delta, trigger_size_sum / num_samples, trigger_distortion_sum / num_samples

    def check(self, model, dataloader, delta, target, exp_vect=None):
        model.eval() 
        device = next(model.parameters()).device
        correct = 0
        total = 0
        with torch.no_grad():  
            for x, y in dataloader:
                if exp_vect is None:
                    x = torch.zeros(x.size())

                x = x.to(device)
                x_adv = torch.add(x, delta)
                target_tensor = torch.full(y.size(), target, device=device)
                pred = model(x_adv)
                correct += (pred.argmax(1) == target_tensor).type(torch.int).sum().item()
                total += y.size(0)
            accuracy = correct / total * 100
            print('target = {}, attack success rate = {}'.format(target, accuracy))
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
                "train_dataset": datasets.MNIST('./data', train=True, download=False, transform=self.mnist_transform),
                "test_dataset": datasets.MNIST('./data', train=False, transform=self.mnist_transform),
                "input_size": (28, 28),
                "last_size": 128
            },
            "CIFAR-10": {
                "train_dataset": datasets.CIFAR10('./data', train=True, download=False, transform=self.cifar10_transform),
                "test_dataset": datasets.CIFAR10('./data', train=False, transform=self.cifar10_transform),
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
            sub_model = SubMNIST()

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
        truncated_targets = test_dataset.targets[:len(last_layer_test_dataset)]
        last_layer_test_dataset = TensorDataset(torch.Tensor(np.array(last_layer_test_dataset)),
                                                torch.Tensor(np.array(truncated_targets)))
        last_layer_test_dataloader = DataLoader(last_layer_test_dataset, **self.test_kwargs)
        return test_dataloader, last_layer_test_dataloader

    def evaluate_triggers(self, model, dataloader, target_lst, size, exp_vect=None):
        acc_lst, dist0_lst, dist2_lst = [], [], []

        for target in target_lst:
            delta, trigger_size, trigger_distortion = self.generate_trigger(model, dataloader, size, target, self.norm, exp_vect, minx=0.0, maxx=1.0)
            delta = torch.where(abs(delta) < 0.1, 0, delta)
            acc = self.check(model, dataloader, delta, target, exp_vect)
            dist0 = torch.norm(delta, 0)
            dist2 = torch.norm(delta, 2)
            acc_lst.append(acc)
            dist0_lst.append(dist0.detach().item())
            dist2_lst.append(dist2.detach().item())
            print('trigger_size = {}, trigger_distortion = {}'.format(trigger_size, trigger_distortion))
            print('target = {}, delta = {}, dist0 = {}, dist2 = {}\n'.format(target, delta[:10], dist0, dist2))
        return delta, acc_lst, dist0_lst, dist2_lst

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
        acc_th = np.percentile(acc_lst, acc_percent)
        mad = np.median(np.abs(dist0_lst - np.median(dist0_lst)))
        if mad != 0.0:
            ano_lst = np.abs(dist0_lst - np.median(dist0_lst)) / mad
            ano_th = np.percentile(ano_lst, ano_percent)
        else:
            ano_th = np.inf

        d0_th = np.percentile(dist0_lst, d0_percent)
        d2_th = np.percentile(dist2_lst, d2_percent)
        print('acc_th = {}, ano_th = {}, d0_th = {}, d2_th = {}\n'.format(acc_th, ano_th, d0_th, d2_th))
        for target, (acc, ano, d0, d2) in enumerate(zip(acc_lst, ano_lst, dist0_lst, dist2_lst)):
            if acc_lst[target] >= acc_th and (ano_th == np.inf or ano_lst[target] <= ano_th or dist0_lst[target] <= d0_th or dist2_lst[target] <= d2_th):
                detected_backdoors.append(target)
        return detected_backdoors
    
    def calculate_true_positive_rate_false_positive_rate(self, thresholds, true_positives, true_negatives, false_positives, false_negatives):
        true_positive_rate_list = []
        false_positive_rate_list = []

        for threshold in thresholds:
            true_positive_rate = true_positives / (true_positives + false_negatives)
            false_positive_rate = false_positives / (false_positives + true_negatives)
            true_positive_rate_list.append(true_positive_rate)
            false_positive_rate_list.append(false_positive_rate)

        return true_positive_rate_list, false_positive_rate_list
    
    @staticmethod
    def model_generator(clean_root_dir, backdoor_root_dir):
        for clean_subdir, _, files in os.walk(clean_root_dir):
            for file in files:
                if file == "model.pt":
                    model_path = os.path.join(clean_subdir, file)
                    yield model_path, "clean"

        for backdoor_subdir, _, files in os.walk(backdoor_root_dir):
            for file in files:
                if file == "model.pt":
                    model_path = os.path.join(backdoor_subdir, file)
                    yield model_path, "backdoor"

    def solve(self, model, assertion, display=None):
        total_true_positives, total_true_negatives, total_false_positives, total_false_negatives = 0, 0, 0, 0
        patch_true_positives, patch_false_negatives, blended_true_positives, blended_false_negatives = 0, 0, 0, 0
        print(f"Experiment Stats - acc_th={acc_percent}, ano_th={ano_percent}, d0_th={d0_percent}, d2_th={d2_percent}, lamb = {lamb}, lr = {lr}")
        
        for model_path, model_type in self.model_generator(clean_root_dir, backdoor_root_dir):
            start_time = time.time()
            print(model_type.capitalize())
            
            model, sub_model, dataset, train_dataset, test_dataset, last_layer = self.load_and_prepare_model(model_path)
            test_dataloader, last_layer_test_dataloader = self.get_last_layer_activations(model, test_dataset, last_layer, dataset)
            target_lst = range(10)
            delta, acc_lst, dist0_lst, dist2_lst = self.evaluate_triggers(sub_model, last_layer_test_dataloader, target_lst, self.size_last)
            detected_backdoors = self.detect_backdoors(acc_lst, dist0_lst, dist2_lst)

            if model_type == "clean":
                if detected_backdoors:
                    _, acc_lst, dist0_lst, dist2_lst = self.evaluate_triggers(model, test_dataloader, detected_backdoors, self.size_input, delta)
                    detected_backdoors = self.detect_backdoors(acc_lst, dist0_lst, dist2_lst)
                    if detected_backdoors:
                        print("(Wrong) Detected backdoors at targets:", detected_backdoors)
                        total_false_positives += 1
                    else:
                        print("No backdoors detected.")
                        total_true_negatives += 1
                else:
                    print("No backdoors detected.")
                    total_true_negatives += 1

            elif model_type == "backdoor":
                trigger_type = self.load_info(model_path)["trigger_type"]
                if detected_backdoors:
                    _, acc_lst, dist0_lst, dist2_lst = self.evaluate_triggers(model, test_dataloader, detected_backdoors, self.size_input, delta)
                    detected_backdoors = self.detect_backdoors(acc_lst, dist0_lst, dist2_lst)
                    if detected_backdoors:
                        print("Detected backdoors at targets:", detected_backdoors)
                        if trigger_type == "patch":
                            patch_true_positives += 1
                        elif trigger_type == "blended":
                            blended_true_positives += 1
                    else:
                        print("(Wrong) No backdoors detected.")
                        if trigger_type == "patch":
                            patch_false_negatives += 1
                        elif trigger_type == "blended":
                            blended_false_negatives += 1
                else:
                    print("(Wrong) No backdoors detected.")
                    if trigger_type == "patch":
                        patch_false_negatives += 1
                    elif trigger_type == "blended":
                        blended_false_negatives += 1
            
            end_time = time.time()
            print("Elapsed time:", end_time - start_time, "seconds")
            print(f"\n{'*' * 100}\n")
        
        print(f"Experiment Stats - acc_th={acc_percent}, ano_th={ano_percent}, d0_th={d0_percent}, d2_th={d2_percent}, lamb = {lamb}, lr = {lr}")

        # Calculate metrics for patch backdoors
        patch_precision = patch_true_positives / (patch_true_positives + total_false_positives) if patch_true_positives + total_false_positives > 0 else 0
        patch_recall = patch_true_positives / (patch_true_positives + patch_false_negatives) if patch_true_positives + patch_false_negatives > 0 else 0
        patch_f1_score = 2 * patch_precision * patch_recall / (patch_precision + patch_recall) if patch_precision + patch_recall > 0 else 0

        # Calculate metrics for blended backdoors
        blended_precision = blended_true_positives / (blended_true_positives + total_false_positives) if blended_true_positives + total_false_positives > 0 else 0
        blended_recall = blended_true_positives / (blended_true_positives + blended_false_negatives) if blended_true_positives + blended_false_negatives > 0 else 0
        blended_f1_score = 2 * blended_precision * blended_recall / (blended_precision + blended_recall) if blended_precision + blended_recall > 0 else 0

        # Print the calculated metrics
        print(f"Patch backdoors - Precision: {patch_precision:.2f}, Recall: {patch_recall:.2f}, F1-score: {patch_f1_score:.2f}")
        print(f"Blended backdoors - Precision: {blended_precision:.2f}, Recall: {blended_recall:.2f}, F1-score: {blended_f1_score:.2f}")
        print(f"Patch_true_positives: {patch_true_positives:.2f}, Blended_true_positives: {blended_true_positives:.2f}, Patch_false_negatives: {patch_false_negatives:.2f}, Blended_false_negatives: {blended_false_negatives:.2f}")
        
        total_true_positives = patch_true_positives + blended_true_positives
        total_false_negatives = patch_false_negatives + blended_false_negatives

        thresholds = [0.5] 
        true_positive_rate_list, false_positive_rate_list = self.calculate_true_positive_rate_false_positive_rate(thresholds, total_true_positives, total_true_negatives, total_false_positives, total_false_negatives)
        print(f'true_positive_rate_list: {true_positive_rate_list}, false_positive_rate_list: {false_positive_rate_list}')

        end_time = time.time()
        precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        accuracy = (total_true_positives + total_true_negatives) / (total_true_positives + total_true_negatives + total_false_positives + total_false_negatives)
        print(f"total_true_positives: {total_true_positives:.2f}, total_true_negatives: {total_true_negatives:.2f}, total_false_positives: {total_false_positives:.2f}, total_false_negatives: {total_false_negatives:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}, Accuracy: {accuracy:.2f}")
