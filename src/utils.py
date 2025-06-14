import os
import random
import numpy as np
import torch
import pickle
from tqdm import tqdm
from foma import foma, foma_hard
from torchvision import datasets, transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import random_split, Subset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok, epochs):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0

    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        labels_true = labels

        if augment == "Default":  
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)
        
        elif augment == "Mixup":
            images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Manifold-Mixup":
            preds, y_a, y_b, lam = model(images, labels, device, augment, aug_ok=True)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            
        elif augment == "FOMA" or augment == "FOMA_knn_input":
            images, soft_labels = foma(images, labels, num_classes, alpha=1.0, rho=0.9)
            preds = model(images, labels, device, augment, aug_ok, num_classes=num_classes)
            loss  = criterion(preds, soft_labels)
            
        elif augment == "FOMA_latent_random" or "FOMA_knn_latent":
            preds, soft_labels = model(images, labels, device, augment, aug_ok=True, num_classes=num_classes)
            loss = criterion(preds, soft_labels)
        
        elif augment == "FOMA_curriculum":
            if epochs < 100:
                images, soft_labels = foma(images, labels, num_classes, alpha=1.0, rho=0.9)
                preds = model(images, labels, device, augment, aug_ok, num_classes=num_classes)
                loss  = criterion(preds, soft_labels)
            else:
                preds, soft_labels = model(images, labels, device, augment, aug_ok=True, num_classes=num_classes)
                loss = criterion(preds, soft_labels) 
        
        elif augment == "Mixup-Original":
            if epochs < 200:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok)
                loss  = criterion(preds, labels)

        elif augment == "PCA":
            if epochs < 100:
                preds = model(images, labels, device, augment, aug_ok=False)
                loss  = criterion(preds, labels)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)

        elif augment == "Mixup-PCA":
            if epochs < 200:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)
        
        elif augment == "Mixup-Original&PCA":
            if epochs < 200:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_score(labels_true.cpu().detach().numpy(), preds.argmax(dim=-1).cpu().numpy())

    train_loss /= len(train_loader)
    train_acc  /= len(train_loader)
    return train_loss, train_acc

def val(model, val_loader, criterion, device, augment, aug_ok):
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)

            val_loss += loss.item()
            val_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device, augment, aug_ok):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)

            test_loss += loss.item()
            test_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())

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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def local_pca_perturbation(data, device, k=10, alpha=1.0, perturb_prob=1.0):
    """
    局所PCAに基づく摂動をデータに加える（近傍の散らばり内に収める）
    :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
    :param device: 使用するデバイス（cuda or cpu）
    :param k: k近傍の数
    :param alpha: 摂動の強さ（最大主成分の標準偏差に対する割合）
    :return: 摂動後のテンソル（同shape）
    """
    data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    N, D = data_np.shape
    if N < k:
        k = N

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
    _, indices = nbrs.kneighbors(data_np)

    perturbed_data = np.copy(data_np)

    for i in range(N):
        if random.random() < perturb_prob:
            neighbors = data_np[indices[i]]
            pca = PCA(n_components=min(D, k))
            pca.fit(neighbors)
            components = pca.components_           # shape: (n_components, D)
            variances = pca.explained_variance_    # shape: (n_components,)

            # ノイズベクトル（各主成分方向に沿った合成）
            noise = np.zeros(D)
            for j in range(len(components)):
                    noise += np.random.randn() * np.sqrt(variances[j]) * components[j]

            # ノイズの方向はそのまま、長さをスケールする
            if np.linalg.norm(noise) > 0:
                noise = noise / np.linalg.norm(noise)

            # 局所の最大主成分の標準偏差に比例したスケール
            max_std = np.sqrt(variances[0])  # 最大分散方向
            scaled_noise = alpha * max_std * noise
            perturbed_data[i] += scaled_noise
        
        else:
            pass

    return torch.tensor(perturbed_data, dtype=torch.float32).to(device)

def make_helix(n_samples):
    n_samples = 5000
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    data = np.vstack((x, y, z)).T
    color = t
    return data, color

def make_spiral(n_samples):
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    data = np.vstack((x, y, z)).T
    color = t
    return data, color

def generate_high_dim_data(regressors, low_dim_data):
    high_dim_data = np.zeros((low_dim_data.shape[0], len(regressors)))
    for i, regressor in enumerate(regressors):
        high_dim_data[:, i] = regressor.predict(low_dim_data)
    return high_dim_data