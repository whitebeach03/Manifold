import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset, ConcatDataset
from tqdm import tqdm
from src.models.mlp import MLP
from src.models.cnn import SimpleCNN
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from src.models.resnet_hidden import ResNet18_hidden, ResNet34_hidden, ResNet50_hidden, ResNet101_hidden, ResNet152_hidden
from src.utils import *
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def evaluate_model(model, dataloader, device, augment):
    model.eval()
    top1_correct = 0
    top3_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if augment == "mixup_hidden":
                outputs = model(images, labels, mixup_hidden=False)
            else:
                outputs = model(images, labels, device, augment, aug_ok=False)
            _, top1_pred = outputs.topk(1, dim=1)
            _, top3_pred = outputs.topk(3, dim=1)
            top1_correct += (top1_pred.squeeze() == labels).sum().item()
            top3_correct += sum([labels[i] in top3_pred[i] for i in range(labels.shape[0])])
            total += labels.size(0)
    top1_error = 1 - (top1_correct / total)
    top3_error = 1 - (top3_correct / total)
    return top1_error, top3_error

def main():
    model_type = 'resnet18'
    data_type = 'stl10'
    epochs = 20
    # augmentations = ["Original", "Flipping", "Cropping", "Rotation", "Translation", "Noisy", "Blurring", "Random-Erasing"]
    augmentations = ["perturb", "pca"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    test_dataset = torchvision.datasets.STL10(root='./data', split='train',  transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    for augment in augmentations:
        print(f" Test {augment} method now.")
        if augment == "mixup_hidden":
            model = ResNet18_hidden().to(device)
        else:
            model = ResNet18().to(device)
        
        model_save_path = f'./logs/{model_type}/Fine-Tuning/{augment}/{data_type}_{epochs}.pth'
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()
        top1_error, top3_error = evaluate_model(model, test_loader, device, augment)
        print(f'{augment} -> Top-1 Error: {top1_error:.2%}, Top-3 Error: {top3_error:.2%}')
        
        pickle_file_path = f'./history/{model_type}/Fine-Tuning/{augment}/{data_type}_{epochs}_test.pickle'
        with open(pickle_file_path, 'rb') as f:
            history = pickle.load(f)
        print("{:.2f}".format(history["acc"]*100), "{:.2f}".format(history["loss"]))

if __name__ == '__main__':
    main()