import os
import random
import numpy as np
import torch
import pickle
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import random_split, Subset, DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomCrop
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.distributions import Beta
import math
from src.methods.mix_methods import *   
from scipy.special import betaincinv
from src.methods.mixup import *
from src.methods.cutout import Cutout
from src.methods.cutmix import cutmix_data
from src.methods.augmix import AugMixTransform
from src.methods.pca import compute_almp_loss_wrn
from src.methods.svd import compute_almp_loss_svd
from src.methods.foma import compute_foma_loss
from src.methods.fomix import *
from src.methods.hybrid import *

def train(model, train_loader, criterion, optimizer, device, augment, num_classes, epochs):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    total_epochs = 250
    t_mixup = int(total_epochs * 0.9)
    
    mixup_fn   = Mixup(alpha=1.0, mode="batch", num_classes=num_classes)
    skmixup_fn = KernelMixup(alpha=1.0, mode="batch", num_classes=num_classes, warping="beta_cdf", tau_max=1.0, tau_std=0.25, lookup_size=4096,)
    confmix_fn = DynamicMixupShuffled(alpha_max=1.0, alpha_min=0.1, conf_th=0.7)

    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        labels_true = labels

        if augment == "Default":  
            preds = model(images, labels, device, augment)
            loss  = criterion(preds, labels)

        elif augment == "ResizeMix":
            images, y_a, y_b, lam = resizemix_data(images, labels, alpha=1.0, device=device)
            preds = model(images, labels, device, augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "SaliencyMix":
            images, y_a, y_b, lam = saliencymix_data(model, images, labels, criterion, alpha=1.0, device=device)
            preds = model(images, labels, device, augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Mixup":
            images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds = model(images, labels, device, augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Manifold-Mixup":
            preds, y_a, y_b, lam = model(images, labels, device, augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            
        elif augment == "CutMix":
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            preds = model(images, labels, device, augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
        elif augment == "RegMixup":
            preds = model(images, labels=labels, device=device, augment=augment)
            loss_clean = criterion(preds, labels)
            
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds_mix = model(mixed_x, labels, device, augment)
            loss_mix = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
            loss = loss_clean + loss_mix
    
        elif augment=="SK-Mixup":
            with torch.no_grad():
                feats = model.extract_features(images)

            mixed_x, mixed_y, k_lam, index = skmixup_fn(images, labels, feats)
            # 3) 元ラベル
            y_a = labels
            y_b = labels[index]
            # 4) 順伝播 + Mixup 損失
            preds = model(mixed_x, labels=None, device=device, augment=augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, k_lam)

        elif augment == "Ent-Mixup":
            images, y_a, y_b, lam, alpha = ent_augment_mixup(
                x=images,
                y=labels,
                model=model,
                alpha_max=1.0,     # 例: 1.0
                num_classes=num_classes
            )
            preds = model(images, labels, device, augment)
            loss  = mixup_criterion(criterion, preds, y_a, y_b, lam)

        # elif augment == "Local-FOMA":
        #     mixed_x, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
        #     preds_mix = model(mixed_x, labels, device, augment)
        #     loss_mix = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
            
        #     phase2_epoch = epochs
        #     phase2_total = 40  
        #     w_foma = min(1.0, phase2_epoch / (phase2_total / 2))
        #     loss_foma, preds = compute_foma_loss(model, images, labels, k=k_foma, num_classes=num_classes, lambda_almp=w_foma, device=device, scaleup=False)
        #     w_mix = 1 - w_foma
            
        #     loss = w_mix*loss_mix + loss_foma

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        y_pred = preds.argmax(dim=1)
        batch_acc = (y_pred == labels_true).float().mean().item()
        train_acc += batch_acc

    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)
    return train_loss, train_acc

def val(model, val_loader, criterion, device, augment):
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, test=True)
            loss  = criterion(preds, labels)

            val_loss += loss.item()
            y_pred = preds.argmax(dim=1)
            batch_acc = (y_pred == labels).float().mean().item()
            val_acc += batch_acc

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device, augment):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, test=True)
            loss  = criterion(preds, labels)

            test_loss += loss.item()
            y_pred = preds.argmax(dim=1)
            batch_acc = (y_pred == labels).float().mean().item()
            test_acc += batch_acc

    test_loss /= len(test_loader)
    test_acc  /= len(test_loader)
    return test_loss, test_acc

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    loss = lam * loss_a + (1 - lam) * loss_b
    return loss.mean()  

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

