import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
from collections import OrderedDict
import random
import time


# Define the MaskedConv2d module
class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        self.mask = Parameter(torch.Tensor(self.weight.size()))
        self.noise = Parameter(torch.Tensor(self.weight.size()))
        init.ones_(self.mask)
        init.zeros_(self.noise)
        self.is_perturbed = False
        self.is_masked = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.noise, a=-eps, b=eps)
        else:
            init.zeros_(self.noise)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def include_mask(self):
        self.is_masked = True

    def exclude_mask(self):
        self.is_masked = False

    def require_false(self):
        self.mask.requires_grad = False
        self.noise.requires_grad = False

    def forward(self, input):
        if self.is_perturbed:
            weight = self.weight * (self.mask + self.noise)
        elif self.is_masked:
            weight = self.weight * self.mask
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# Define the MNIST network with MaskedConv2d
class MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_Network, self).__init__()
        self.conv1 = MaskedConv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d(16, 32, 4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = MaskedConv2d(32, 32, 4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_state_dict(net, orig_state_dict):
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


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def Regularization(model):
    L1 = 0
    L2 = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            L1 += torch.sum(torch.abs(param))
            L2 += torch.norm(param, 2)
    return L1, L2


def mask_train(model, criterion, mask_opt, data_loader, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0

    for _, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # Calculate the adversarial perturbation for images
        ori_lab = torch.argmax(model(images), axis=1).long()
        perturbed_images = images + args['eps'] * torch.sign(torch.randn_like(images))

        # Forward pass
        output_noise = model(perturbed_images)
        output_clean = model(images)

        # Calculate loss
        loss_rob = criterion(output_noise, ori_lab)
        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        # Add regularization terms to the loss
        loss = args['alpha'] * loss_nat + (1 - args['alpha']) * loss_rob + args['beta'] * L1 + args['gamma'] * L2

        # Update model
        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

        # Calculate accuracy
        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
        total_loss += loss.item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(model, criterion, data_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)

    return loss, acc

def main():
    # Set up the configurations and hyperparameters
    args = {
        'batch_size': 128,
        'eps': 0.4,
        'alpha': 0.5,
        'beta': 0.001,
        'gamma': 0.001,
    }

    # Set random seed for reproducibility
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=args['batch_size'], shuffle=False)

    # Create the model
    model = MNIST_Network().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # Perform mask training
            loss, acc = mask_train(model, criterion, optimizer, trainloader, args)

            running_loss += loss
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{10}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100}")
                running_loss = 0.0

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss, test_acc = test(model, criterion, testloader)
        print(f"Epoch [{epoch + 1}/{10}], Test Loss: {test_loss}, Test Accuracy: {test_acc}")


if __name__ == '__main__':
    main()