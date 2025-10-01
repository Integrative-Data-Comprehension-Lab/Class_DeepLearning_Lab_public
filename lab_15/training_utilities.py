import os, time, shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_cifar10_dataloaders(data_root_dir, device, batch_size, num_worker):
    validation_size = 0.2
    random_seed = 42

    normalize = transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) 
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=train_transforms)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(root=data_root_dir, train=False, download=True, transform=test_transforms)

    num_classes = len(train_dataset.classes)

    # Split train dataset into train and validataion dataset
    train_indices, val_indices = train_test_split(np.arange(len(train_dataset)), 
                                                  test_size=validation_size, random_state=random_seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # DataLoader
    kwargs = {}
    if device.startswith("cuda"):
        kwargs.update({
            'pin_memory': True,
        })

    train_dataloader = DataLoader(dataset = train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_worker, **kwargs)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size=batch_size, sampler=valid_sampler,
                                num_workers=num_worker, **kwargs)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False, 
                                 num_workers=num_worker, **kwargs)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes


def train_one_epoch(model, device, dataloader, criterion, optimizer, epoch):
    """ train for one epoch """
    model.train() # switch to train mode

    total_loss = 0.0
    total_samples = 0

    dataloader_tqdm = tqdm(dataloader, desc = f'Training Epoch {epoch + 1}', total = len(dataloader), leave=False)
    for X, y in dataloader_tqdm:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.shape[0]
        total_samples += X.shape[0]
        
        dataloader_tqdm.set_postfix({"loss": f"{loss.item():.4e}"})

    dataloader_tqdm.close()

    average_loss = total_loss / total_samples
    return average_loss


def evaluate_one_epoch(model, device, dataloader, criterion, epoch = 0):
    """ Evaluate the model for one epoch"""
    model.eval() # switch to evaluate mode

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        dataloader_tqdm = tqdm(dataloader, desc = 'Validation/Test', total = len(dataloader), leave=False)
        for X, y in dataloader_tqdm:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.shape[0]
            total_samples += X.shape[0]

            y_pred = logits.argmax(dim = 1)
            total_correct += (y_pred == y).sum().item()

            dataloader_tqdm.set_postfix({"loss": f"{loss.item():.4e}"})

        dataloader_tqdm.close()

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return average_loss, accuracy