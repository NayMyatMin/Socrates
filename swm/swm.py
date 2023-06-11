import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
from collections import OrderedDict
from utils import *
import random
import time

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


def load_model(model_class, name, *args):
    
    model = model_class(*args)
    model.load_state_dict(torch.load(name).state_dict())
    
    return model


def main():
    # Set up the configurations and hyperparameters
    args = {
        'batch_size': 128,
        'eps': 0.4, # Magnitude of the adversarial perturbation 
        'alpha': 0.9, # Balance between the natural loss and the robust loss 
        'beta': 1e-3, # L1 regularization term
        'gamma': 1e-2, # L2 regularization term
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

    model_path = "../dataset/2M-Backdoor/id-0378/model.pt"
    state_dict = torch.load(model_path, map_location=device)
    net = MNIST_Network().to(device)
    load_state_dict(net, orig_state_dict=state_dict.state_dict())
    model = net.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)

    best_test_acc = 0.0  # Variable to track the best test accuracy
    best_model_path = "best_model.pt"  # Path to save the best model

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

        # Save the model if the current test accuracy is the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

    print("Training complete!")
    print(f"Best Test Accuracy: {best_test_acc}")
    print(f"Best Model saved at: {best_model_path}")

if __name__ == '__main__':
    main()