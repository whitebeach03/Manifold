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

def main():
    iteration  = 3
    epochs     = 250
    data_type  = "cifar100"
    model_type = "wide_resnet_28_10"
    augmentations = ["Original", "Mixup", "Mixup-Original", "Mixup-PCA"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print()
    for augment in augmentations:
        print(augment)
        pickle_file_path = f"history/{model_type}/{augment}/{data_type}_{epochs}"
        avg_acc  = load_acc(pickle_file_path, iteration)
        avg_loss = load_loss(pickle_file_path, iteration)
        std_acc  = load_std(pickle_file_path, iteration)

        print(f"| Test Accuracy: {avg_acc}% | Test Loss: {avg_loss} | std: {std_acc} |")
        print()

def load_acc(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    avg_acc = 0
    for i in range(iteration):
        avg_acc += dic[i]["acc"]
    avg_acc = avg_acc / iteration
    avg_acc *= 100
    avg_acc = round(avg_acc, 2)
    return avg_acc

def load_loss(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    avg_loss = 0
    for i in range(iteration):
        avg_loss += dic[i]["loss"]
    avg_loss = avg_loss / iteration
    avg_loss = round(avg_loss, 3)
    return avg_loss

def load_std(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    acc_list = []
    for i in range(iteration):
        acc_list.append(dic[i]["acc"])
    std_acc = np.std(acc_list)
    return std_acc


if __name__ == '__main__':
    main()