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

def train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok, epochs, k_foma=32):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    history = {"alpha": []}
    total_epochs = 250
    t_mixup = int(total_epochs * 0.9)
    
    mixup_fn   = Mixup(alpha=1.0, mode="batch", num_classes=num_classes)
    skmixup_fn = KernelMixup(alpha=1.0, mode="batch", num_classes=num_classes, warping="beta_cdf", tau_max=1.0, tau_std=0.25, lookup_size=4096,)
    confmix_fn = DynamicMixupShuffled(alpha_max=1.0, alpha_min=0.1, conf_th=0.7)

    batch_idx = 0
    for images, labels in tqdm(train_loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        labels_true = labels

        if augment == "Default":  
            preds = model(images, labels, device, augment, aug_ok)
            loss  = criterion(preds, labels)
        
        elif augment == "FMix":
            # FMix: 周波数領域でのマスク生成
            # decay_power=3.0 は自然画像の統計性質に合致するため推奨
            images, index, lam = fmix_data(images, alpha=1.0, decay_power=3.0, shape=images.shape[2:])
            y_a, y_b = labels, labels[index]
            
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "ResizeMix":
            # ResizeMix: 画像Bをリサイズしてペースト
            images, y_a, y_b, lam = resizemix_data(images, labels, alpha=1.0, device=device)
            
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "SaliencyMix":
            # SaliencyMix: 勾配を利用して重要領域をカット＆ペースト
            # 内部で勾配計算を行うため、optimizer.zero_grad()のタイミングに注意が必要だが、
            # 以下の処理はパラメータ更新用の勾配には影響させず、入力画像の勾配のみを使用する。
            images, y_a, y_b, lam = saliencymix_data(model, images, labels, criterion, alpha=1.0, device=device)
            
            # Saliency計算のための計算グラフはここで一度切れているため、再度Forwardして学習用グラフを作る
            # （SaliencyMix内ではモデルをEvalモードで使ったりdetachしているため安全）
            model.train() # 念のためTrainモード保証
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "PuzzleMix":
            # PuzzleMix: Saliency情報を利用してパッチの混合を最適化
            # 計算コストが高いため、ここでは Greedy Block Matching 戦略を採用
            images, y_a, y_b, lam = puzzlemix_data(model, images, labels, criterion, alpha=1.0, device=device, block_size=4)
            
            model.train()
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
        elif augment == "ES-Mixup":
            if epochs < t_mixup:
                images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds = model(images, labels, device, augment, aug_ok)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            else:
                preds = model(images, labels, device, augment, aug_ok)
                loss  = criterion(preds, labels)

        elif augment == "Mixup":
            images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Manifold-Mixup":
            preds, y_a, y_b, lam = model(images, labels, device, augment, aug_ok=True)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            
        elif augment == "CutMix":
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
        elif augment == "RegMixup":
            preds = model(images, labels=labels, device=device, augment=augment)
            loss_clean = criterion(preds, labels)
            
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds_mix = model(mixed_x, labels, device, augment, aug_ok)
            loss_mix = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
            loss = loss_clean + loss_mix
            
        elif augment == "Local-FOMA":
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds_mix = model(mixed_x, labels, device, augment, aug_ok)
            loss_mix = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
            
            phase2_epoch = epochs
            phase2_total = 40  
            w_foma = min(1.0, phase2_epoch / (phase2_total / 2))
            loss_foma, preds = compute_foma_loss(model, images, labels, k=k_foma, num_classes=num_classes, lambda_almp=w_foma, device=device, scaleup=False)
            w_mix = 1 - w_foma
            
            loss = w_mix*loss_mix + loss_foma
            
        elif augment == "Mixup-FOMA2":
            if epochs < t_mixup:
                preds = model(images, labels=labels, device=device, augment=augment)
                loss_clean = criterion(preds, labels)
                mixed_x, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
                preds_mix = model(mixed_x, labels, device, augment, aug_ok)
                loss_mix = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
                loss = loss_clean + loss_mix
            else:
                if num_classes == 100:
                    total_epochs = 400
                elif num_classes == 10:
                    total_epochs = 250
                phase2_epoch = epochs - t_mixup
                phase2_total = total_epochs - t_mixup  # Phase2全体の長さ
                # 前半は係数を上げる、後半は1.0で固定
                w_foma = min(1.0, phase2_epoch / (phase2_total / 2))
                loss, preds = compute_foma_loss(model, images, labels, k=10, num_classes=num_classes, lambda_almp=w_foma, device=device, scaleup=False)
    
        elif augment=="SK-Mixup":
            with torch.no_grad():
                feats = model.extract_features(images)
            # 1) λ, index を取得
            lam, index = skmixup_fn._get_params(images.size(0), device)
            # 2) SK-Mixup で画像を合成
            mixed_x, _ = skmixup_fn(images, labels, feats)
            # 3) 元ラベル
            y_a = labels
            y_b = labels[index]
            # 4) 順伝播 + Mixup 損失
            preds = model(mixed_x, labels=None, device=device, augment=augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)


        elif augment == "Conf-Mixup":
            preds = model(images, labels=labels, device=device, augment=augment)
            loss_clean = criterion(preds, labels)
            # 予測信頼度（最大ソフトマックス確率）を取得
            with torch.no_grad():
                logits = model(images, labels=labels, device=device, augment=augment)           # -> (B, C)
                probs  = F.softmax(logits, dim=1)
                # labels が (B,) の LongTensor だとすると…
                conf = probs[torch.arange(probs.size(0)), labels]  # -> (B,)
                # print(conf)
            mixed_x, mixed_y = confmix_fn(images, labels, conf, num_classes)
            preds = model(mixed_x, labels=labels, device=device, augment=augment)
            mix_loss = -(mixed_y * F.log_softmax(preds, dim=1)).sum(dim=1).mean()
            loss = mix_loss

        elif augment == "Ent-Mixup":
            images, y_a, y_b, lam, alpha = ent_augment_mixup(
                x=images,
                y=labels,
                model=model,
                alpha_max=1.0,     # 例: 1.0
                num_classes=num_classes
            )
            preds = model(images, labels, device, augment, aug_ok)
            loss  = mixup_criterion(criterion, preds, y_a, y_b, lam)
            history["alpha"].append(alpha)

        batch_idx += 1
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
            y_pred = preds.argmax(dim=1)
            batch_acc = (y_pred == labels).float().mean().item()
            val_acc += batch_acc

    val_loss /= len(val_loader)
    val_acc  /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader, criterion, device, augment, aug_ok):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            
            preds = model(images, labels, device, augment, aug_ok)
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

