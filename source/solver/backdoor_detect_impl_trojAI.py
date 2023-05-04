import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.lib_layers import *
from model.lib_models import *
from nn_utils import *
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from utils import *
from wrn import WideResNet

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
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)  # Add this line to flatten the input tensor
        x = self.fc(x)
        return x

class ModelLoader:
    def __init__(self, device):
        self.device = device
        self.mnist_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
        self.cifar10_transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

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

    def load_info(self, model_path):
        try:
            with open(os.path.join(os.path.dirname(model_path), "info.json")) as f:
                info = json.load(f)
                return info
        except FileNotFoundError as e:
            print(f"Error loading info.json: {e}")
            raise

class Metrics:
    @staticmethod
    def calculate_true_positive_rate_false_positive_rate(true_positives, true_negatives, false_positives, false_negatives):
        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)
        print(f'True_positive_rate: {true_positive_rate}, False_positive_rate: {false_positive_rate}')

    @staticmethod
    def calculate_patch_blended_metrics(true_positives, false_positives, false_negatives):
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1_score

    @staticmethod
    def print_backdoor_metrics(patch_metrics, blended_metrics, patch_counts, blended_counts):
        (patch_precision, patch_recall, patch_f1_score), (blended_precision, blended_recall, blended_f1_score) = patch_metrics, blended_metrics
        (patch_true_positives, patch_false_negatives), (blended_true_positives, blended_false_negatives) = patch_counts, blended_counts
        print(f"Patch backdoors - Precision: {patch_precision:.2f}, Recall: {patch_recall:.2f}, F1-score: {patch_f1_score:.2f}")
        print(f"Blended backdoors - Precision: {blended_precision:.2f}, Recall: {blended_recall:.2f}, F1-score: {blended_f1_score:.2f}")
        print(f"Patch_true_positives: {patch_true_positives:.2f}, Blended_true_positives: {blended_true_positives:.2f}, Patch_false_negatives: {patch_false_negatives:.2f}, Blended_false_negatives: {blended_false_negatives:.2f}")

    @staticmethod
    def calculate_metrics(patch_true_positives, patch_false_negatives, blended_true_positives, blended_false_negatives, total_false_positives, total_true_negatives):
        patch_metrics = Metrics.calculate_patch_blended_metrics(patch_true_positives, total_false_positives, patch_false_negatives)
        blended_metrics = Metrics.calculate_patch_blended_metrics(blended_true_positives, total_false_positives, blended_false_negatives)
        patch_counts = (patch_true_positives, patch_false_negatives)
        blended_counts = (blended_true_positives, blended_false_negatives)
        total_true_positives = patch_true_positives + blended_true_positives
        total_false_negatives = patch_false_negatives + blended_false_negatives
        Metrics.calculate_true_positive_rate_false_positive_rate(total_true_positives, total_true_negatives, total_false_positives, total_false_negatives)
        Metrics.print_backdoor_metrics(patch_metrics, blended_metrics, patch_counts, blended_counts)
        precision, recall, f1_score, accuracy = Metrics.calculate_overall_metrics(total_true_positives, total_false_positives, total_false_negatives, total_true_negatives)
        Metrics.print_overall_metrics(total_true_positives, total_true_negatives, total_false_positives, total_false_negatives, precision, recall, f1_score, accuracy)

    @staticmethod
    def calculate_precision_recall_f1_score(true_positives, false_positives, false_negatives):
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1_score

    @staticmethod
    def calculate_overall_metrics(total_true_positives, total_false_positives, total_false_negatives, total_true_negatives):
        precision, recall, f1_score = Metrics.calculate_precision_recall_f1_score(total_true_positives, total_false_positives, total_false_negatives)
        accuracy = (total_true_positives + total_true_negatives) / (total_true_positives + total_true_negatives + total_false_positives + total_false_negatives)
        return precision, recall, f1_score, accuracy

    @staticmethod
    def print_overall_metrics(total_true_positives, total_true_negatives, total_false_positives, total_false_negatives, precision, recall, f1_score, accuracy):
        print(f"Total_true_positives: {total_true_positives:.2f}, Total_true_negatives: {total_true_negatives:.2f}, Total_false_positives: {total_false_positives:.2f}, Total_false_negatives: {total_false_negatives:.2f}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}, Accuracy: {accuracy:.2f}")

class AttackSpecification:
    @staticmethod
    def load_and_display_attack_specification(file_path):
        attack_specification = torch.load(file_path)
        for key, value in attack_specification.items():
            if key == 'target_label':
               print(f"true_{key}: {value}\n")
        return attack_specification

alphabet = 'M'; ano_th, acc_th = -2, 50; lamb, lr = 1, 0.1
clean_root_dir = f"dataset/1{alphabet}-Benign"; backdoor_root_dir = f"dataset/10{alphabet}-Backdoor"

class BackdoorDetectHiddenImpl:
    def check_class(self, model, exp_vect):
        pred = model(exp_vect.unsqueeze(0))
        _, predicted_class = torch.max(pred, 1)
        return predicted_class.item()
    
    def optimize_exp_vector(self, model, exp_vect, target, device, lr, lamb, patience, tolerance, max_iterations):
        optimizer = optim.Adam([exp_vect], lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience, eps=tolerance)
        iteration = 0
        best_loss = float('inf')
        no_improvement_count = 0
        best_exp_vect = exp_vect.clone().detach().requires_grad_(True)
        best_d0 = float('inf')

        while iteration < max_iterations:
            with torch.cuda.amp.autocast():
                pred = model(exp_vect.unsqueeze(0))
                target_tensor = torch.tensor([target], device=device, dtype=torch.long)
                loss = F.cross_entropy(pred, target_tensor) + lamb * torch.norm(exp_vect, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                exp_vect.data = torch.relu(exp_vect.data)

            scheduler.step(loss.item())
            if loss.item() < best_loss - tolerance:
                best_loss = loss.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count >= patience:
                break
            predicted_class = self.check_class(model, exp_vect)

            if predicted_class == target:
                curr_d0 = torch.count_nonzero(exp_vect).item()
                if curr_d0 < best_d0:
                    best_d0 = curr_d0
                    best_exp_vect = exp_vect.clone().detach().requires_grad_(True)

            iteration += 1
        return best_exp_vect, self.check_class(model, best_exp_vect)

    def generate_hidden_layer_exp_vector(self, model, size, target_lst, max_iterations=10000, patience=100, tolerance=1e-4):
        torch.manual_seed(7)
        model.eval()
        exp_vect_lst, result = [], []
        device = next(model.parameters()).device
        exp_vect = torch.randn(size, device=device) * 0.1
        # exp_vect = torch.zeros(size, device=device)
        with open('trigger_activation.txt', 'a') as f:
            for target in target_lst:
                exp_vect = exp_vect.clone().detach().requires_grad_(True)
                optimized_exp_vect, predicted_class = self.optimize_exp_vector(
                    model, exp_vect, target, device, lr=0.1, lamb=1, patience=patience, tolerance=tolerance, max_iterations=max_iterations)
                # print(f"target: {target} | exp_vect: \n{optimized_exp_vect[:10]}, dist0: {torch.count_nonzero(optimized_exp_vect)}")
                exp_vect_lst.append(optimized_exp_vect)
                if predicted_class == target:
                    result.append(f'Target: {target}, Predicted Class: {predicted_class}, Match: True')
                else:
                    result.append(f'Target: {target}, Predicted Class: {predicted_class}, Match: False')
                    f.write(f"target: {target} | exp_vect: \n{optimized_exp_vect}, dist0: {torch.count_nonzero(optimized_exp_vect)}\n")    
        return exp_vect_lst, result
    
    def evaluate_triggers_with_exp_vect_lst(self, model, exp_vect_lst, num_classes):
        dist0_lst = []
        device = next(model.parameters()).device

        for exp_vect in exp_vect_lst:
            exp_vect = exp_vect.to(device)
            dist0 = torch.count_nonzero(exp_vect).item()

            # Compute and store the L0 distance for each exp_vect
            dist0_lst.append(dist0)

        return dist0_lst

class BackdoorDetectInputImpl:

    def second_submodel(self, model):
        if isinstance(model, MNIST_Network):
            sub_model_layers = list(model.main.children())[:-1]
        elif isinstance(model, WideResNet):
            sub_model_layers = list(model.children())[:-1]
        else:
            raise ValueError("Unsupported model type.")
        sub_model = nn.Sequential(*sub_model_layers)
        return sub_model
 
    def get_clamp(self, epoch, batch, max_clamp, min_clamp, num_batches, num_epochs):
        t = epoch * num_batches + batch
        T = num_batches * num_epochs
        clamp = (max_clamp - min_clamp) * 0.5 * (1 + np.cos(np.pi * t / T)) + min_clamp
        return clamp

    def get_min_max_values(dataset, device):
        if dataset == 'M':
            mean = torch.tensor([0.1307], device=device)
            std = torch.tensor([0.3081], device=device)
        elif dataset == 'C':
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
            std = torch.tensor([0.2023, 0.1994, 0.2010], device=device)
        else:
            raise ValueError("Unsupported dataset type.")

        minx = (-mean / std).to(device)
        maxx = ((1 - mean) / std).to(device)
        return minx, maxx

    def check(self, model, dataloader, delta, target, exp_vect=None):
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
            accuracy = correct / total * 100
            print('target = {}, attack success rate = {}'.format(target, accuracy))
        return correct / total
    
    def evaluate_triggers(self, model, second_submodel, dataloader, target_lst, size, exp_vect_lst):
        acc_lst, dist2_lst = [], []
        for target in target_lst:
            exp_vect = exp_vect_lst[target_lst.index(target)]
            delta = self.generate_trigger(model, second_submodel, dataloader, size, target, exp_vect)
            delta = torch.where(abs(delta) < 0.01, 0, delta)
            acc = self.check(model, dataloader, delta, target, exp_vect)
            acc_lst.append(acc)
            
            dist2 = torch.norm(delta, 2)
            dist2_lst.append(dist2.detach().item())
            print('delta = {}, dist2 = {}\n'.format(delta[:10], dist2))

        return target_lst, acc_lst, dist2_lst

    def generate_trigger(self, model, second_submodel, dataloader, size, target, exp_vect, minx=None, maxx=None, patience=5, min_improvement=1e-4, num_of_epochs=100):
        device = next(model.parameters()).device
        delta = torch.randn(size, device=device) * 0.1
        delta.requires_grad = True
        best_delta, best_loss = None, float('inf')
        optimizer = optim.Adam([delta], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)
        scaler = GradScaler()

        if minx is None or maxx is None:
            minx, maxx = BackdoorDetectInputImpl.get_min_max_values(alphabet, device)

        for epoch in range(num_of_epochs):
            # Train the second_submodel
            for batch, (x, y) in enumerate(dataloader):
                x = x.to(device)
                delta.requires_grad = True
                with torch.cuda.amp.autocast():
                    # Generate adversarial examples by adding delta to the input
                    x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                    pred = second_submodel(x_adv)

                    # Filter out large neurons using a mask based on the threshold
                    large_neurons_threshold = 0.4
                    large_neurons_mask = (exp_vect.view(-1, 1).expand(-1, pred.shape[1]) > large_neurons_threshold).float()
                    lamb = 0.01
                    # Calculate the MSE loss for large neurons and add L2-norm regularization
                    exp_vect_expanded = exp_vect.view(-1, 1).expand(-1, pred.shape[1])
                    mse_large_neurons = F.mse_loss(pred * large_neurons_mask, exp_vect_expanded * large_neurons_mask)
                    loss = mse_large_neurons + lamb * torch.norm(delta, 2)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Evaluate the trigger on the main model
            model.eval()
            trigger_quality, trigger_size, trigger_distortion, num_samples = 0, 0, 0, 0
            with torch.no_grad():
                for batch, (x, y) in enumerate(dataloader):
                    x = x.to(device); y = y.to(device)
                    x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                    preds = model(x_adv)
                    num_samples += x.size(0)

                    # Calculate trigger success, size, and distortion
                    trigger_success = (torch.argmax(pred, dim=1) == target).sum().item()
                    trigger_quality += trigger_success
                    trigger_size += torch.norm(delta.view(delta.size(0), -1), 2, dim=1).sum().item()
                    trigger_distortion += torch.norm(delta, 2).item() * x.size(0)

                current_loss = trigger_quality / num_samples
                scheduler.step(current_loss)

            # Update best_delta and best_loss if there is significant improvement
            if best_loss - current_loss > min_improvement:
                best_loss = current_loss
                best_delta = delta.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_delta
    
    def detect_backdoors(self, dist_lst, target_lst, detection_type, acc_lst=None):
        detected_backdoors = []
        epsilon = 1e-9
        med = np.median(dist_lst)
        dev_lst = np.abs(dist_lst - med)
        mad = np.median(dev_lst)
        ano_lst = (dist_lst - med) / (mad + epsilon)
        ano_th, acc_th = -2, 50

        print(f"Accuracy list for all targets_{detection_type}: {acc_lst}")

        if detection_type == "hidden":
            d_th = 0.1 * 128 # num_neurons
        elif detection_type == "input":
            input_neuron = 784 if alphabet == 'M' else 3072 if alphabet == 'C' else None
            d_th = 0.1 * input_neuron # 28 * 28 = 784, 32 * 32 * 3 = 3072
        else:
            raise ValueError(f"Invalid detection_type: {detection_type}")

        print(f"Distance list_{detection_type}: {dist_lst}")
        print(f"Median of the distance list_{detection_type}: {med}")

        for i, (ano, d, tgt) in enumerate(zip(ano_lst, dist_lst, target_lst)):
            if acc_lst is not None:
                acc = acc_lst[i]
            if detection_type == "hidden" or (detection_type == "input" and acc >= acc_th):
                if ano <= ano_th or d <= d_th:
                    detected_backdoors.append(tgt)

        print(f"Absolute deviations from the median_{detection_type}: {dev_lst}")
        print(f"Median absolute deviation (MAD)_{detection_type}: {mad}")
        print(f"Anomaly scores_{detection_type}: {ano_lst}")  
        return detected_backdoors

class BackdoorDetectImpl:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.train_kwargs = {'batch_size': 100, 'num_workers': 8, 'pin_memory': True, 'drop_last':False}
        self.test_kwargs = {'batch_size': 128, 'num_workers': 8, 'pin_memory': True, 'drop_last':True}
        self.size_input = None; self.size_last = None

    def model_generator(self, clean_root_dir, backdoor_root_dir):
        for clean_subdir, _, files in os.walk(clean_root_dir):
            for file in files:
                if file == "model.pt":
                    model_path = os.path.join(clean_subdir, file)
                    yield model_path, "clean", None

        for backdoor_subdir, _, files in os.walk(backdoor_root_dir):
            for file in files:
                if file == "model.pt":
                    model_path = os.path.join(backdoor_subdir, file)
                    attack_spec_path = os.path.join(backdoor_subdir, "attack_specification.pt")
                    yield model_path, "backdoor", attack_spec_path
           
    def apply_trigger(self, x, trigger, mask, alpha, top_left, bottom_right):
        trigger_pattern = trigger.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        mask_pattern = mask.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        sliced_mask_pattern = mask_pattern[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        sliced_trigger_pattern = trigger_pattern[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        x[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = x[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] * (1 - sliced_mask_pattern) + sliced_trigger_pattern * sliced_mask_pattern * alpha
        return x

    def analyze_hidden_layer_activations(self, sub_model, test_dataloader, last_layer, last_layer_test_dataloader, attack_specification):
        sub_model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Collect trigger-embedded input activations
        with open('trigger_activation.txt', 'a') as f:
            with torch.no_grad():
                for _, (x, y) in enumerate(last_layer_test_dataloader):
                    x = x.to(device)
                    sub_model(x)
                    trigger_activation = torch.flatten(activation[last_layer], 1).cpu().detach().numpy()
                    
                    # Write trigger activation for current input to file
                    f.write(f"Attack specification\n")
                    f.write(f'{attack_specification}\n')
                    f.write(f"Trigger activation\n")
                    np.savetxt(f, trigger_activation, fmt='%.6f', delimiter=',')

    def get_last_layer_activations(self, model, test_dataset, last_layer, dataset, attack_specification, apply_trigger_ratio=0.1):
        model.eval()
        test_dataloader = torch.utils.data.DataLoader(test_dataset, **self.test_kwargs)
        last_layer_test_dataset = []
        trigger = attack_specification['trigger']['pattern']
        mask = attack_specification['trigger']['mask']
        alpha = attack_specification['trigger']['alpha']
        top_left = attack_specification['trigger']['top_left']
        bottom_right = attack_specification['trigger']['bottom_right']
        
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_dataloader):
                # Apply trigger to a portion of the input data
                trigger_indices = np.random.choice(len(x), int(len(x) * apply_trigger_ratio), replace=False)
                x[trigger_indices] = self.apply_trigger(x[trigger_indices], trigger, mask, alpha, top_left, bottom_right)

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

    def process_clean_model(self, detected_backdoors_hidden, model, second_submodel, test_dataloader, exp_vect_lst, model_loader, total_true_negatives, total_false_positives):
        input = BackdoorDetectInputImpl()
        if detected_backdoors_hidden:
            target_lst, acc_lst, dist2_lst = input.evaluate_triggers(model, second_submodel, test_dataloader, detected_backdoors_hidden, model_loader.size_input, exp_vect_lst)
            detected_backdoors_input = input.detect_backdoors(dist2_lst, target_lst, "input", acc_lst)
            if detected_backdoors_input:
                print("(Wrong) Detected backdoors at targets:", detected_backdoors_input)
                total_false_positives += 1
            else:
                print("No backdoors detected.")
                total_true_negatives += 1
        else:
            print("No backdoors detected.")
            total_true_negatives += 1
        return total_true_negatives, total_false_positives

    def process_backdoor_model(self, detected_backdoors_hidden, model, second_submodel, test_dataloader, exp_vect_lst, model_loader, attack_spec_path, trigger_type, 
    patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives):
        input = BackdoorDetectInputImpl()
        if detected_backdoors_hidden:
            target_lst, acc_lst, dist2_lst = input.evaluate_triggers(model, second_submodel, test_dataloader, detected_backdoors_hidden, model_loader.size_input, exp_vect_lst)
            detected_backdoors_input = input.detect_backdoors(dist2_lst, target_lst, "input", acc_lst)
            if detected_backdoors_input:
                print("Detected backdoors at targets:", detected_backdoors_input)
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
        return patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives

    def solve(self, model, assertion, display=None):
        total_true_positives, total_true_negatives, total_false_positives, total_false_negatives = 0, 0, 0, 0
        patch_true_positives, patch_false_negatives, blended_true_positives, blended_false_negatives = 0, 0, 0, 0
        model_loader = ModelLoader(self.device); metrics = Metrics(); attack_specification = AttackSpecification()
        hidden = BackdoorDetectHiddenImpl(); input = BackdoorDetectInputImpl()

        for model_path, model_type, attack_spec_path in self.model_generator(clean_root_dir, backdoor_root_dir):
            start_time = time.time()
            print(model_type.capitalize())
            model, sub_model, dataset, train_dataset, test_dataset, last_layer = model_loader.load_and_prepare_model(model_path)
            second_submodel = input.second_submodel(model)

            target_lst = range(10)
            exp_vect_lst, result = hidden.generate_hidden_layer_exp_vector(sub_model, model_loader.size_last, target_lst)
            dist0_lst = hidden.evaluate_triggers_with_exp_vect_lst(model, exp_vect_lst, len(target_lst))
            detected_backdoors_hidden = input.detect_backdoors(dist0_lst, target_lst, "hidden")
            print("Suspected Target List:", detected_backdoors_hidden)
            #print(*result, sep="\n")
        
            if model_type == "clean":
                total_true_negatives, total_false_positives = self.process_clean_model(detected_backdoors_hidden, model, second_submodel, test_dataloader, exp_vect_lst, model_loader, total_true_negatives, total_false_positives)
            elif model_type == "backdoor":
                attack_spec = attack_specification.load_and_display_attack_specification(attack_spec_path)
                test_dataloader, last_layer_test_dataloader = self.get_last_layer_activations(model, test_dataset, last_layer, dataset, attack_spec)
                self.analyze_hidden_layer_activations(sub_model, test_dataloader, last_layer,last_layer_test_dataloader, attack_spec)
                trigger_type = model_loader.load_info(model_path)["trigger_type"]
                patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives = self.process_backdoor_model(detected_backdoors_hidden, model, second_submodel, test_dataloader, exp_vect_lst, model_loader, attack_spec_path, trigger_type, patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives)

            end_time = time.time()
            print("Elapsed time:", end_time - start_time, "seconds")
            print(f"\n{'*' * 100}\n")
        
        print(f"Experiment Stats - acc_th_percent={acc_th}, ano_th={ano_th}, lamb = {lamb}, lr = {lr}")
        metrics.calculate_metrics(patch_true_positives, patch_false_negatives, blended_true_positives, blended_false_negatives, total_false_positives, total_true_negatives)