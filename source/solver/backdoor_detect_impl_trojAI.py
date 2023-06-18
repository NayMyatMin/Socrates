import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.lib_layers import *
from model.lib_models import *
from nn_utils import *
import numpy
import autograd.numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from utils import *
from swm import *
from wrn import WideResNet
from torchvision.utils import save_image

alphabet = 'M'; ano_th, acc_th = -2, 50; lamb, lr = 1, 0.1
clean_root_dir = f"dataset/2{alphabet}-Benign"; backdoor_root_dir = f"dataset/2{alphabet}-Backdoor"

class ModelLoader:

    def __init__(self, device):
        self.device = device
        self.mnist_transform = transforms.ToTensor()
        self.cifar10_transform = transforms.ToTensor()


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
   

    def load_and_prepare_model(self, model_path, SWM=None):
        info = self.load_info(model_path)
        dataset = info["dataset"]
        dataset_config = self.get_dataset(dataset)
        
        if dataset_config is None:
            raise ValueError(f"Invalid dataset name: {dataset}")

        model = None
        if SWM != None:
            model = load_model(New_MNIST_Network, model_path)
        elif dataset == "MNIST":
            model = load_model(MNIST_Network, model_path)
        elif dataset == "CIFAR-10":
            model = load_model(WideResNet, model_path)

        if model is not None:
            model.to(self.device)

        self.size_input = dataset_config["input_size"]
        self.size_last = dataset_config["last_size"]

        return model, dataset, dataset_config["train_dataset"], dataset_config["test_dataset"]


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
    def load_attack_specification(file_path):
        attack_specification = torch.load(file_path)
        return attack_specification



class BackdoorDetectHiddenImpl:

    def get_submodel(self, model):
        if isinstance(model, MNIST_Network) | isinstance(model, New_MNIST_Network):
            sub_model_layer = list(model.main.children())[-1]
        elif isinstance(model, WideResNet):
            sub_model_layer = list(model.children())[-1]
        else:
            raise ValueError("Unsupported model type.")
        sub_model = nn.Sequential(sub_model_layer)
        return sub_model

    
    def print_active_weights(self, model, exp_vect):
        # Check that model is a Sequential type
        if not isinstance(model, nn.Sequential):
            raise Exception("Expected the model to be of type 'Sequential'.")

        # Get the first (and in this case, only) module from the Sequential model
        linear_layer = next(iter(model.named_children()))[1]

        # Check that the module is a Linear layer
        if not isinstance(linear_layer, nn.Linear):
            raise Exception("Expected the module to be of type 'nn.Linear'.")

        weight_matrix = linear_layer.weight.detach().cpu().numpy()
        active_indices = torch.nonzero(exp_vect, as_tuple=True)[0].cpu().numpy()

        for idx in active_indices:
            print(f"Active Neuron: {idx}, Weights: {weight_matrix[:, idx]}\n")

    def optimize_exp_vector(self, model, exp_vect, size, target, device, lr, lamb, patience, tolerance, max_iterations):
        optimizer = optim.Adam([exp_vect], lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience, eps=tolerance)
        
        iteration = 0; no_improvement_count = 0; norm = 2
        best_loss = float('inf'); best_d0 = float('inf')
        best_exp_vect = torch.zeros(size)

        target_tensor = torch.tensor([target], device=device, dtype=torch.long)

        for i in range(size):
            cand_vect = torch.zeros(size, device=device)
            cand_vect[i] = 10.0

            pred = model(cand_vect.unsqueeze(0))
            loss = F.cross_entropy(pred, target_tensor)

            if torch.argmax(pred).item() == target:
                if loss < best_loss:
                    best_loss = loss
                    best_exp_vect = cand_vect

        if best_loss == float('inf'):
            for i in range(size):
                for j in range(size):
                    if i != j:
                        cand_vect = torch.zeros(size, device=device)
                        cand_vect[i] = 10.0; cand_vect[j] = 10.0

                        pred = model(cand_vect.unsqueeze(0))
                        loss = F.cross_entropy(pred, target_tensor)

                        if torch.argmax(pred).item() == target:
                            if loss < best_loss:
                                best_loss = loss
                                best_exp_vect = cand_vect
            if best_loss == float('inf'):
                assert False
        
        print(best_loss)
        print(list(best_exp_vect.to('cpu').numpy()))

        # print('iteration = {}'.format(iteration))
        return best_exp_vect, torch.argmax(model(best_exp_vect)).item()


    def generate_exp_vector(self, model, test_dataloader, size, target_lst, max_iterations=10000, patience=100, tolerance=1e-4):
        torch.manual_seed(7)
        model.eval(); device = next(model.parameters()).device
        exp_vect_lst, result = [], []
        
        for target in target_lst:
            exp_vect = torch.randn(size, device=device) * 0.1
            exp_vect = exp_vect.detach().requires_grad_(True)

            optimized_exp_vect, predicted_class = self.optimize_exp_vector(
                model, exp_vect, size, target, device, lr=0.1, lamb=1e1, patience=patience, tolerance=tolerance, max_iterations=max_iterations)
            self.print_active_weights(model, optimized_exp_vect)
            exp_vect_lst.append(optimized_exp_vect)

            if predicted_class == target:
                print(f'Target: {target}, Predicted Class: {predicted_class}, Match: True')
            else:
                print(f'Target: {target}, Predicted Class: {predicted_class}, Match: False')
                # assert False

        return exp_vect_lst, result
    

    def evaluate_exp_vect_lst(self, exp_vect_lst):
        dist0_lst = []

        for exp_vect in exp_vect_lst:
            # Compute and store the L0 distance for each exp_vect
            dist0 = torch.count_nonzero(exp_vect).item()
            dist0_lst.append(dist0)

        return dist0_lst



class BackdoorDetectInputImpl:

    def get_submodel(self, model):
        if isinstance(model, MNIST_Network) | isinstance(model, New_MNIST_Network):
            sub_model_layers = list(model.main.children())[:-1]
        elif isinstance(model, WideResNet):
            sub_model_layers = list(model.children())[:-1]
        else:
            raise ValueError("Unsupported model type.")
        sub_model = nn.Sequential(*sub_model_layers)
        return sub_model


    def get_min_max_values(self):
        return 0.0, 1.0


    def check(self, model, dataloader, delta, target, exp_vect=None):
        model.eval(); device = next(model.parameters()).device
        correct = 0; total = 0
        minx, maxx = BackdoorDetectInputImpl().get_min_max_values()

        with torch.no_grad():  
            for x, y in dataloader:
                x = x.to(device)
                x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                target_tensor = torch.full(y.size(), target, device=device)

                pred = model(x_adv)
                correct += (pred.argmax(1) == target_tensor).type(torch.int).sum().item()
                total += y.size(0)

            accuracy = correct / total * 100
            print('target = {}, attack success rate = {}'.format(target, accuracy))

        return correct / total

    
    def evaluate_triggers(self, model, input_submodel, dataloader, target_lst, size, exp_vect_lst):
        acc_lst, dist2_lst, dist0_lst = [], [], []

        for target in target_lst:
            exp_vect = exp_vect_lst[target_lst.index(target)]
            
            delta, succ = self.generate_trigger(model, input_submodel, dataloader, size, target, exp_vect)
            # delta = torch.where(abs(delta) < 0.01, 0, delta)
            
            acc = self.check(model, dataloader, delta, target, exp_vect)
            acc_lst.append(acc)
            
            dist2 = torch.norm(delta) 
            dist0 = torch.norm(delta, 0)  # type: ignore
            dist2_lst.append(dist2.detach().item())
            dist0_lst.append(dist0.detach().item())
            print('delta 365 = {}, dist0 = {}, dist2 = {}\n'.format(delta, dist0, dist2))

        return target_lst, acc_lst, dist2_lst, dist0_lst


    def generate_trigger_cw2(self, model, input_submodel, dataloader, size, target, exp_vect, fixed_lst=None, patience=5, min_improvement=1e-4, num_of_epochs=10, print_gradient=False):
        device = next(model.parameters()).device
        delta = torch.zeros(size, device=device)
        delta.requires_grad = True
        best_delta, best_succ = None, 0.0

        optimizer = optim.Adam([delta], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

        minx, maxx = BackdoorDetectInputImpl().get_min_max_values()
        patience_counter = 0; lamb = 0.01
        for epoch in range(num_of_epochs):
            for batch, (x, y) in enumerate(dataloader): 
                x = x.to(device)
                delta.requires_grad = True
                # Generate adversarial examples by adding delta to the input
                x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                pred = input_submodel(x_adv)
                exp_vect_expanded = exp_vect.repeat(x.size(0), 1) 
                mse_large_neurons = F.mse_loss(pred, exp_vect_expanded)
                loss = mse_large_neurons + lamb * torch.norm(delta)

                optimizer.zero_grad()
                loss.backward() 

                if fixed_lst is not None:
                    for (i, j) in fixed_lst:
                        if delta.grad is not None:
                            delta.grad.data[i,j] = 0.0

                if print_gradient and epoch == 0 and batch == 0:
                        print(f'Gradient at the first epoch: {delta.grad}')

                optimizer.step() # Gradient update

            # Evaluate the trigger on the main model
            model.eval()
            trigger_quality, trigger_size, trigger_distortion, num_samples = 0, 0, 0, 0
            with torch.no_grad():
                for batch, (x, y) in enumerate(dataloader):
                    x = x.to(device); y = y.to(device)
                    x_adv = torch.clamp(torch.add(x, delta), minx, maxx)
                    pred = model(x_adv)
                    num_samples += x.size(0)

                    # Calculate trigger success, size, and distortion
                    trigger_success = (torch.argmax(pred, dim=1) == target).sum().item()
                    trigger_quality += trigger_success
                    trigger_size += torch.norm(delta.view(delta.size(0), -1), dim=1).sum().item()
                    trigger_distortion += torch.norm(delta).item() * x.size(0)

                current_succ = trigger_quality / num_samples
                scheduler.step(current_succ)

            # Update best_delta and best_loss if there is significant improvement
            if best_delta is None or current_succ > best_succ:
                best_succ = current_succ
                best_delta = delta.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        assert best_delta is not None
        return best_delta, best_succ


    def generate_trigger(self, model, input_submodel, dataloader, size, target, exp_vect, fixed_lst=None, num_of_epochs=10):
        fixed_lst = []; best_adv = None
        fixed_num = 784; fixed_idx = numpy.full(fixed_num, False)
        old_fixed_lst = []; old_fixed_idx = numpy.full(fixed_num, False)
        best_delta, best_succ = None, 0.0
        device = next(model.parameters()).device
        print(f'Number of Epochs = {num_of_epochs}')
        count = 0
        while fixed_num > 0:
            if count == 0:  
                delta, succ = self.generate_trigger_cw2(model, input_submodel, dataloader, size, target, exp_vect, fixed_lst, print_gradient=True)
            else:
                delta, succ = self.generate_trigger_cw2(model, input_submodel, dataloader, size, target, exp_vect, fixed_lst, print_gradient=False)
            
            count += 1
            if best_delta is None or succ > 0.8:
                old_fixed_lst = fixed_lst.copy()
                old_fixed_idx = fixed_idx.copy()
                best_delta = delta; best_succ = succ
            else:
                fixed_num = fixed_num // 2
                fixed_lst = old_fixed_lst.copy()
                fixed_idx = old_fixed_idx.copy()
                delta = best_delta; succ = best_succ

            delta_lst = enumerate(list(torch.flatten(torch.abs(delta)).to('cpu').numpy()))
            delta_sorted = sorted(delta_lst, key = lambda x: x[-1])

            cnt = 0
            for i in range(784):
                idx, val = delta_sorted[i]
                if not fixed_idx[idx]:
                    fixed_lst.append((idx // 28, idx % 28))
                    fixed_idx[idx] = True
                    cnt += 1
                if cnt == fixed_num:
                    break

        if best_delta is not None:
            best_delta[abs(best_delta) < 0.0001] = 0.0

            # Choose 10 images to create adversarial examples
            images = []
            dataloader_iter = iter(dataloader)
            for _ in range(1):
                img, _ = next(dataloader_iter) 
                img = img.to(device)
                images.append(img)

            minx, maxx = BackdoorDetectInputImpl().get_min_max_values()
            images_adv = [torch.clamp(torch.add(img, best_delta), minx, maxx) for img in images]
            images = [(img - minx) / (maxx - minx) for img in images]
            images_adv = [(img_adv - minx) / (maxx - minx) for img_adv in images_adv]

            # Concatenate each pair of original and adversarial images along width dimension
            pairs = [torch.cat((img[0], img_adv[0]), 2) for img, img_adv in zip(images, images_adv)]
            all_images = torch.stack(pairs, dim=0)
            save_image(all_images, f"images/{model_path.split('/')[1]}_{model_path.split('/')[-2]}_{model_path.split('/')[-1]}_target_{target}.png")

        return best_delta, best_succ
    


class BackdoorDetectImpl:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_kwargs = {'batch_size': 100, 'num_workers': 8, 'pin_memory': True, 'drop_last': False}
        self.test_kwargs = {'batch_size': 100, 'num_workers': 8, 'pin_memory': True, 'drop_last': False}
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
        
           

    def get_dataloaders(self, input_submodel, test_dataset, dataset):
        input_submodel.eval()
        test_dataset.data = test_dataset.data[:1000]
        test_dataset.targets = test_dataset.targets[:1000]
        test_dataloader = DataLoader(test_dataset, **self.test_kwargs)
        print("Test dataset size: ", len(test_dataset))
        hidden_test_dataset = []
    
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_dataloader):
                x = x.to(self.device)
                if dataset == 'MNIST':
                    hidden_test_dataset.extend(input_submodel(x).cpu().detach().numpy())
                elif dataset == 'CIFAR-10':
                    hidden_test_dataset.extend(input_submodel(x).cpu().detach().numpy())
        
        hidden_test_dataset = TensorDataset(torch.Tensor(np.array(hidden_test_dataset)),
                                                torch.Tensor(np.array(test_dataset.targets)))
        hidden_test_dataloader = DataLoader(hidden_test_dataset, **self.test_kwargs)
        return test_dataloader, hidden_test_dataloader


    def detect_backdoors(self, dist_lst, target_lst, detection_type, acc_lst=None, dist0_lst=None):
        detected_backdoors = []
        epsilon = 1e-9
        med = numpy.median(dist_lst)
        dev_lst = numpy.abs(dist_lst - med)
        mad = numpy.median(dev_lst)
        ano_lst = (dist_lst - med) / (mad + epsilon)
        ano_th, acc_th = -2, 0.5

        print(f"Accuracy list for all targets_{detection_type}: {acc_lst}")

        if detection_type == "hidden":
            d_th = 0.1 * 128 # num_neurons
        elif detection_type == "input": # 28 * 28 = 784, 32 * 32 * 3 = 3072
            d_th = 0.1 * 784 if alphabet == 'M' else 0.1 * 3072 if alphabet == 'C' else None
        else:
            raise ValueError(f"Invalid detection_type: {detection_type}")

        print(f"Distance list_{detection_type}: {dist_lst}")
        if dist0_lst is not None:
            print(f"Distance list0_{detection_type}: {dist0_lst}")
        print(f"Median of the distance list_{detection_type}: {med}")
        print(f"Absolute deviations from the median_{detection_type}: {dev_lst}")
        print(f"Median absolute deviation (MAD)_{detection_type}: {mad}")
        print(f"Anomaly scores_{detection_type}: {ano_lst}")

        for i, (ano, d, tgt) in enumerate(zip(ano_lst, dist_lst, target_lst)):
            acc = None 
            if acc_lst is not None:
                acc = acc_lst[i]
            if detection_type == "hidden" or (detection_type == "input" and acc is not None and acc >= acc_th):
                if ano <= ano_th or d <= d_th:
                    detected_backdoors.append(tgt)

        return detected_backdoors


    def process_clean_model(self, detected_backdoors_hidden, model, input_submodel, test_dataloader, exp_vect_lst, model_loader, total_true_negatives, total_false_positives):
        input = BackdoorDetectInputImpl()
        if detected_backdoors_hidden:
            target_lst, acc_lst, dist2_lst, dist0_lst = input.evaluate_triggers(model, input_submodel, test_dataloader, detected_backdoors_hidden, model_loader.size_input, exp_vect_lst)
            detected_backdoors_input = self.detect_backdoors(dist2_lst, target_lst, "input", acc_lst, dist0_lst)
        
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

    
    def process_backdoor_model(self, detected_backdoors_hidden, model, input_submodel, test_dataloader, exp_vect_lst, model_loader, attack_spec_path, trigger_type, 
    patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives):
        input = BackdoorDetectInputImpl()

        if detected_backdoors_hidden:
            target_lst, acc_lst, dist2_lst, dist0_lst = input.evaluate_triggers(model, input_submodel, test_dataloader, detected_backdoors_hidden, model_loader.size_input, exp_vect_lst)
            detected_backdoors_input = self.detect_backdoors(dist2_lst, target_lst, "input", acc_lst, dist0_lst)
        
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


    def apply_trigger(self, x, attack_spec):
        trigger = attack_spec['trigger']
        pattern, mask, alpha = trigger['pattern'], trigger['mask'], trigger['alpha']
        
        pattern = pattern.to(self.device); mask = mask.to(self.device)

        x = mask * (alpha * pattern + (1 - alpha) * x) + (1 - mask) * x
        return x


    def test_attack_hypothesis(self, model, input_submodel, test_dataloader, attack_spec):
        model.eval(); input_submodel.eval()
        hidden_vect_lst = []
        target = attack_spec['target_label']

        print('target = {}'.format(target))
        
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_dataloader):
                # Apply trigger to a portion of the input data
                x = self.apply_trigger(x, attack_spec)
                x = x.to(self.device)
                
                h = input_submodel(x)
                pred = torch.argmax(model(x), dim=1)
                
                hidden_vect_lst.extend(list(h[pred == target].cpu().detach().numpy()))
        
        print('success = {}\n'.format(len(hidden_vect_lst)))
        no_neurons = len(hidden_vect_lst[0])
        count = torch.zeros(no_neurons)
        
        for i in range(no_neurons):
            for vect in hidden_vect_lst:
                if vect[i] > 0.0:
                    count[i] += 1

        print('count = {}'.format(list(count)))
        print('common = {}\n'.format(list(count >= 1.0 * len(hidden_vect_lst))))
        print('common = {}\n'.format(list(count >= 1.0 * len(hidden_vect_lst)).count(True)))


    def test_accuracy_hypothesis(self, model, input_submodel, test_dataloader, target):
        model.eval(); input_submodel.eval()
        hidden_vect_lst = []

        print('\n##############\n')
        print('target = {}'.format(target))
        
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_dataloader):
                # Apply trigger to a portion of the input data
                x = x.to(self.device)
                
                h = input_submodel(x)
                pred = torch.argmax(model(x), dim=1)
                
                hidden_vect_lst.extend(list(h[(pred == y) & (y == target)].cpu().detach().numpy()))
        
        print('success = {}\n'.format(len(hidden_vect_lst)))
        no_neurons = len(hidden_vect_lst[0])
        count = torch.zeros(no_neurons)
        
        for i in range(no_neurons):
            for vect in hidden_vect_lst:
                if vect[i] > 0.0:
                    count[i] += 1

        print('count = {}'.format(list(count)))
        print('common = {}\n'.format(list(count >= 1.0 * len(hidden_vect_lst))))
        print('common = {}\n'.format(list(count >= 1.0 * len(hidden_vect_lst)).count(True)))


    def test_accuracy(self, model, input_submodel, hidden_submodel, test_dataloader):
        correct = 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_dataloader):
                x = x.to(self.device)
                assert torch.all(model(x) == hidden_submodel(input_submodel(x)))
                pred = torch.argmax(model(x), dim=1)
                correct += (pred == y).type(torch.int).sum().item()
        print('correct = {}\n'.format(correct))


    def solve(self, model, assertion, display=None):
        total_true_positives, total_true_negatives, total_false_positives, total_false_negatives = 0, 0, 0, 0
        patch_true_positives, patch_false_negatives, blended_true_positives, blended_false_negatives = 0, 0, 0, 0

        model_loader = ModelLoader(self.device); metrics = Metrics(); attack_specification = AttackSpecification()
        hidden = BackdoorDetectHiddenImpl(); input = BackdoorDetectInputImpl()
        global model_path
        for model_path, model_type, attack_spec_path in self.model_generator(clean_root_dir, backdoor_root_dir):
            start_time = time.time()
            
            print('\n##########################################\n')
            print(model_type.capitalize())
            
            # model, dataset, train_dataset, test_dataset = model_loader.load_and_prepare_model(model_path)
            # hidden_submodel = hidden.get_submodel(model)
            # input_submodel = input.get_submodel(model)

            # test_dataloader, hidden_test_dataloader = self.get_dataloaders(input_submodel, test_dataset, dataset)
            # print(f'model_name = {model_path}')

            # target_lst = range(10)
            # exp_vect_lst, result = hidden.generate_exp_vector(hidden_submodel, hidden_test_dataloader, model_loader.size_last, target_lst)
            # dist0_lst = hidden.evaluate_exp_vect_lst(exp_vect_lst)
            # print(dist0_lst)

            # detected_backdoors_hidden = self.detect_backdoors(dist0_lst, target_lst, "hidden")
            # print("Suspected Target List:", detected_backdoors_hidden)

            # if model_type == "clean":
            #     total_true_negatives, total_false_positives = self.process_clean_model(detected_backdoors_hidden, model, input_submodel, test_dataloader, exp_vect_lst, model_loader, total_true_negatives, total_false_positives)
            # elif model_type == "backdoor":
            #     attack_spec = attack_specification.load_attack_specification(attack_spec_path)
            #     print('\nTrue target label = {}\n'.format(attack_spec['target_label']))

            #     trigger_type = model_loader.load_info(model_path)["trigger_type"]
            #     print(f'trigger_type = {trigger_type}')
            #     patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives = self.process_backdoor_model(detected_backdoors_hidden, model, input_submodel, test_dataloader, exp_vect_lst, model_loader, attack_spec_path, trigger_type, patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives)

            # # Soft Weight Masking
            # swm = SWM(model_path)
            # swm.swm()
            
            print('\n##########################################\n')
            swm_model_path = os.path.dirname(model_path) + "/swm_model.pt"
            model, dataset, train_dataset, test_dataset = model_loader.load_and_prepare_model(swm_model_path, SWM=True)
            hidden_submodel = hidden.get_submodel(model)
            input_submodel = input.get_submodel(model)
            test_dataloader, hidden_test_dataloader = self.get_dataloaders(input_submodel, test_dataset, dataset)
            model_path = swm_model_path
            print(f'model_name = {model_path}')
            
            target_lst = range(10)
            exp_vect_lst, result = hidden.generate_exp_vector(hidden_submodel, hidden_test_dataloader, model_loader.size_last, target_lst)
            dist0_lst = hidden.evaluate_exp_vect_lst(exp_vect_lst)
            print(dist0_lst)

            detected_backdoors_hidden = self.detect_backdoors(dist0_lst, target_lst, "hidden")
            print("Suspected Target List:", detected_backdoors_hidden)

            if model_type == "clean":
                total_true_negatives, total_false_positives = self.process_clean_model(detected_backdoors_hidden, model, input_submodel, test_dataloader, exp_vect_lst, model_loader, total_true_negatives, total_false_positives)
            elif model_type == "backdoor":
                attack_spec = attack_specification.load_attack_specification(attack_spec_path)
                print('\nTrue target label = {}\n'.format(attack_spec['target_label']))

                trigger_type = model_loader.load_info(model_path)["trigger_type"]
                print(f'trigger_type = {trigger_type}')
                patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives = self.process_backdoor_model(detected_backdoors_hidden, model, input_submodel, test_dataloader, exp_vect_lst, model_loader, attack_spec_path, trigger_type, patch_true_positives, blended_true_positives, patch_false_negatives, blended_false_negatives)
            
            end_time = time.time()
            print("Elapsed time:", end_time - start_time, "seconds")
            print(f"\n{'*' * 100}\n")
        
        print(f"Experiment Stats - acc_th_percent={acc_th}, ano_th={ano_th}, lamb = {lamb}, lr = {lr}")
        metrics.calculate_metrics(patch_true_positives, patch_false_negatives, blended_true_positives, blended_false_negatives, total_false_positives, total_true_negatives)