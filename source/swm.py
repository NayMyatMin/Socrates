import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
from utils import MNIST_Network
from collections import OrderedDict
import random

class SWM:
    def __init__(self, model_path):
        print("Soft Weight Masking")
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = {
            'batch_size': 128,
            'eps': 0.4,
            'alpha': 0.9,
            'beta': 1e-3,
            'gamma': 1e-2,
        }

    def load_state_dict(self, net, orig_state_dict):
        if 'state_dict' in orig_state_dict.keys():
            orig_state_dict = orig_state_dict['state_dict']
        if "state_dict" in orig_state_dict.keys():
            orig_state_dict = orig_state_dict["state_dict"]

        new_state_dict = OrderedDict()
        for k, v in net.state_dict().items():
            if k in orig_state_dict.keys():
                new_state_dict[k] = orig_state_dict[k]
            elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
                new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
            else:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict)

    def clip_mask(self, model, lower=0.0, upper=1.0):
        params = [param for name, param in model.named_parameters() if 'mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)

    def Regularization(self, model):
        L1 = 0
        L2 = 0
        for name, param in model.named_parameters():
            if 'mask' in name:
                L1 += torch.sum(torch.abs(param))
                L2 += torch.norm(param, 2)
        return L1, L2

    def mask_train(self, model, criterion, mask_opt, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0

        for _, (images, labels) in enumerate(data_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            nb_samples += images.size(0)

            ori_lab = torch.argmax(model(images), axis=1).long()
            perturbed_images = images + self.args['eps'] * torch.sign(torch.randn_like(images))

            output_noise = model(perturbed_images)
            output_clean = model(images)

            loss_rob = criterion(output_noise, ori_lab)
            loss_nat = criterion(output_clean, labels)
            L1, L2 = self.Regularization(model)

            loss = self.args['alpha'] * loss_nat + (1 - self.args['alpha']) * loss_rob + self.args['beta'] * L1 + self.args['gamma'] * L2

            mask_opt.zero_grad()
            loss.backward()
            mask_opt.step()
            self.clip_mask(model)

            pred = output_clean.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum().item()
            total_loss += loss.item()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc

    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    def swm(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=self.args['batch_size'], shuffle=True)
        testloader = DataLoader(testset, batch_size=self.args['batch_size'], shuffle=False)

        state_dict = torch.load(self.model_path, map_location=self.device)
        net = MNIST_Network().to(self.device)
        self.load_state_dict(net, orig_state_dict=state_dict.state_dict())
        model = net.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.9)

        best_test_acc = 0.0
        best_model_path = os.path.dirname(self.model_path) + "/swm_model.pt"

        for epoch in range(1):
            model.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                loss, acc = self.mask_train(model, criterion, optimizer, trainloader)

                running_loss += loss
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{1}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100}")
                    running_loss = 0.0

            model.eval()
            with torch.no_grad():
                test_loss, test_acc = self.test(model, criterion, testloader)
            print(f"Epoch [{epoch + 1}/{10}], Test Loss: {test_loss}, Test Accuracy: {test_acc}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), best_model_path)

        print("Training complete!")
        print(f"Best Test Accuracy: {best_test_acc}")
        print(f"Best Model saved at: {best_model_path}")

if __name__ == '__main__':
    model_path = "dataset/2M-Backdoor/id-0499/model.pt"
    swm = SWM(model_path)
    swm.swm()
