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
from src.models.wide_resnet import Wide_ResNet

def ent_augment_mixup(x, y, model, alpha_max, num_classes, eps=1e-8):
    """
    Mixup インターフェースに合わせて、
    EntAugment による動的 α を使った mixup_data 相当を返す。

    Returns:
      mixed_x: [B, C, H, W]
      y_a:      [B] 元ラベル
      y_b:      [B] シャッフル後ラベル
      lam:      [B] 各サンプルの mixup 比率
    """
    B = x.size(0)

    # 1) 各サンプルのエントロピーに基づき α_i を計算
    with torch.no_grad():
        logits = model(x, labels=None, device=None, augment="Ent-Mixup")                 # [B, num_classes]
        probs  = F.softmax(logits, dim=1)
        ent    = -torch.sum(probs * torch.log(probs + eps), dim=1) / math.log(num_classes)
        mag    = (1.0 - ent).clamp(min=eps)     # 低エントロピーほど大きく
        alpha  = alpha_max * mag               # [B]
        # print(alpha)

    # 2) Beta(alpha_i, alpha_i) から λ_i をサンプル
    lam = Beta(alpha, alpha).sample()         # [B]
    lam = lam.to(x.device)

    # 3) バッチをシャッフル
    idx  = torch.randperm(B).to(x.device)
    x2   = x[idx]
    y2   = y[idx]

    # 4) Mixup 入力生成
    lam_x   = lam.view(B, 1, 1, 1)
    mixed_x = lam_x * x + (1 - lam_x) * x2

    # 5) ラベルペアとして返す
    y_a = y
    y_b = y2

    return mixed_x, y_a, y_b, lam, alpha

class FixedAugmentedDataset(Dataset):
    def __init__(self, base_dataset, random_transform):
        """
        base_dataset:  元の PIL 画像＋ラベルの Dataset
        random_transform: transforms.Compose([...Random...]) のインスタンス
        """
        self.labels = []
        self.images = []
        # ここで一度だけ全サンプルに変換を適用してキャッシュ
        for img, lbl in base_dataset:
            img_aug = random_transform(img)
            self.images.append(img_aug)
            self.labels.append(lbl)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 常にキャッシュ済みの同じ img_aug を返す
        return self.images[idx], self.labels[idx]

# ------------ 可視化用ユーティリティ ------------
# 正規化を解除する関数
def unnormalize(tensor):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)

# バッチをグリッド表示する関数
def visualize_batch(
                    epochs: int,
                    images: torch.Tensor,
                    labels: torch.Tensor=None,
                    augment: str=None, 
                    classes: list=None,
                    n: int = 8,
                    title: str = None):
    """
    images:     Tensor[B, C, H, W], 正規化後の画像
    labels:     Tensor[B]（あるいは list of int）
    classes:    CIFAR-100 クラス名リスト
    n:          表示する枚数（先頭 n 枚）
    title:      画像上部に出したい文字列（デフォルトはラベル名をスペース区切りで）
    """
    images = images.detach().cpu()
    # 先頭 n 枚をアンノーマライズして PIL に戻す
    imgs = [unnormalize(img) for img in images[:n]]
    grid = make_grid(imgs, nrow=n, padding=0)
    
    plt.figure(figsize=(n*1.5, 1.5))
    npimg = grid.numpy().transpose(1,2,0)
    plt.imshow(npimg)
    plt.axis('off')
    
    if labels is not None and classes is not None:
        labs = labels[:n].cpu().numpy()
        lab_str = [classes[l] for l in labs]
        plt.title(title or "  ".join(lab_str), fontsize=10)
    plt.savefig(f"{augment}_{epochs}.png")

def train(model, train_loader, criterion, optimizer, device, augment, num_classes, aug_ok, epochs):
    model.train()
    train_loss = 0.0
    train_acc  = 0.0
    history = {"alpha": []}
    
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
        
        elif augment == "Mixup+FOMA":
            loss, preds = compute_hybrid_loss(model, images, labels)
        
        elif augment == "FOMix":
            loss, preds = compute_fomix_loss(model, images, labels)
        
        elif augment == "FOMA":
            loss, preds = compute_foma_loss(model, images, labels, augment, lambda_almp=1.0, device=device)
        
        elif augment == "FOMA-scaleup":
            loss, preds = compute_foma_loss(model, images, labels, augment, lambda_almp=1.0, device=device)
        
        elif augment == "Local-FOMA":
            loss, preds = compute_foma_loss(model, images, labels, augment, lambda_almp=1.0, device=device)
        
        elif augment == "PCA":
            loss, preds = compute_almp_loss_wrn(model, images, labels, method="pca", lambda_almp=1.0, device=device)
        
        elif augment == "SVD":
            loss, preds = compute_almp_loss_wrn(model, images, labels, method="svd", lambda_almp=1.0, device=device)
        
        elif augment == "Cholesky":
            loss, preds = compute_almp_loss_wrn(model, images, labels, method="cholesky", lambda_almp=1.0, device=device)
        
        elif augment == "CutMix":
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "AugMix":
            # Apply AugMix per image in the batch
            images_aug = torch.zeros_like(images)
            for i in range(images.size(0)):
                images_aug[i] = augmix_transform(images[i])
            preds = model(images_aug, labels, device, augment, aug_ok)
            loss = criterion(preds, labels)
        
        elif augment == "RegMixup":
            preds = model(images, labels=labels, device=device, augment=augment)
            loss_clean = criterion(preds, labels)
            
            lam, index = mixup_fn._get_params(images.size(0), device)
            mixed_x = mixup_fn._linear_mixing(lam, images, index)

            y_a = labels
            y_b = labels[index]
            preds_mix = model(mixed_x, labels=labels, device=device, augment=augment)
            loss_mix = mixup_criterion(criterion, preds_mix, y_a, y_b, lam)
            
            eta = 1.0  
            loss = loss_clean + eta * loss_mix
        
        elif augment=="SK-Mixup":
            # # 1) 特徴量抽出（分類ヘッド直前のベクトル）
            # feats = model.extract_features(images)  # → [B, D]
            # # 2) SK-Mixup 適用
            # mixed_x, mixed_y = skmixup(images, labels, feats)
            # # 3) 順伝播・損失
            # preds = model(mixed_x, labels=None, device=device, augment=augment)  # labels フラグは不要なら外す
            # loss  = criterion(preds, mixed_y)
            
            # (必要なら) まず特徴量を取っておく
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
        
        elif augment == "Teacher-SK-Mixup":
            teacher = Wide_ResNet(28, 10, 0.3, num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load("./logs/wide_resnet_28_10/Mixup/cifar100_400_0.pth", weights_only=True))
            teacher.eval()
            with torch.no_grad():
                feats = teacher.extract_features(images)  # → [B, D]
            lam, index = skmixup_fn._get_params(images.size(0), device)
            mixed_x, _ = skmixup_fn(images, labels, feats)
            y_a = labels
            y_b = labels[index]
            preds = model(mixed_x, labels=None, device=device, augment=augment)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
        elif augment == "Teacher-SK-Mixup-Curriculum":
            if epochs < 50:
                teacher = Wide_ResNet(28, 10, 0.3, num_classes=num_classes).to(device)
                teacher.load_state_dict(torch.load("./logs/wide_resnet_28_10/Mixup/cifar100_400_0.pth", weights_only=True))
                teacher.eval()
                with torch.no_grad():
                    feats = teacher.extract_features(images)  # → [B, D]
            else:
                with torch.no_grad():
                    feats = model.extract_features(images) 
                lam, index = skmixup_fn._get_params(images.size(0), device)
                mixed_x, _ = skmixup_fn(images, labels, feats)
                y_a = labels
                y_b = labels[index]
                preds = model(mixed_x, labels=None, device=device, augment=augment)
                loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
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
        
        elif augment == "Mixup(alpha=0.5)":
            images, y_a, y_b, lam = mixup_data(images, labels, 0.5, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Mixup(alpha=1.0)":
            images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Mixup(alpha=2.0)":
            images, y_a, y_b, lam = mixup_data(images, labels, 2.0, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment == "Mixup(alpha=5.0)":
            images, y_a, y_b, lam = mixup_data(images, labels, 5.0, device)
            preds = model(images, labels, device, augment, aug_ok)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        
        elif augment == "Mixup":
            # images, y_a, y_b, lam = mixup_data(images, labels, 1.0, device)
            # preds = model(images, labels, device, augment, aug_ok)
            # loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            
            # 1) λ と index を取得
            lam, index = mixup_fn._get_params(images.size(0), device)
            # 2) 画像を合成
            mixed_x = mixup_fn._linear_mixing(lam, images, index)
            # 3) 元ラベルを取り出し
            y_a = labels
            y_b = labels[index]
            # 4) 順伝播
            preds = model(mixed_x, labels=None, device=device, augment=augment)
            # 5) 元祖Mixup方式の損失計算
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)

        elif augment.startswith("Manifold-Mixup"):
            preds, y_a, y_b, lam = model(images, labels, device, augment, aug_ok=True)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
            
        elif augment == "FOMA" or augment == "FOMA_knn_input":
            images, soft_labels = foma(images, labels, num_classes, alpha=1.0, rho=0.9)
            preds = model(images, labels, device, augment, aug_ok, num_classes=num_classes)
            loss  = criterion(preds, soft_labels)
        
        elif augment == "FOMA_default":
            images, soft_labels = foma(images, labels, num_classes, alpha=1.0, rho=0.9)
            images = torch.stack([post_foma_transform(img) for img in images], dim=0).to(device)
            preds = model(images, labels, device, augment, aug_ok, num_classes=num_classes)
            loss  = criterion(preds, soft_labels)

        elif augment == "FOMA_latent_random" or "FOMA_knn_latent":
            preds, soft_labels = model(images, labels, device, augment, aug_ok=True, num_classes=num_classes)
            loss = criterion(preds, soft_labels) 

        elif augment == "PCA":
            if epochs < 100:
                preds = model(images, labels, device, augment, aug_ok=False)
                loss  = criterion(preds, labels)
            else:
                preds = model(images, labels, device, augment, aug_ok=True)
                loss  = criterion(preds, labels)

        batch_idx += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # train_acc += accuracy_score(labels_true.cpu().detach().numpy(), preds.argmax(dim=-1).cpu().numpy())
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
            # val_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())
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
            # test_acc  += accuracy_score(labels.cpu().tolist(), preds.argmax(dim=-1).cpu().tolist())
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
        torch.backends.cudnn.deterministic = True

# def local_pca_perturbation(data, device, k=10, alpha=1.0, perturb_prob=1.0):
#     """
#     局所PCAに基づく摂動をデータに加える（近傍の散らばり内に収める）
#     :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
#     :param device: 使用するデバイス（cuda or cpu）
#     :param k: k近傍の数
#     :param alpha: 摂動の強さ（最大主成分の標準偏差に対する割合）
#     :return: 摂動後のテンソル（同shape）
#     """
#     data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
#     N, D = data_np.shape
#     if N < k:
#         k = N

#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
#     _, indices = nbrs.kneighbors(data_np)

#     perturbed_data = np.copy(data_np)

#     for i in range(N):
#         if random.random() < perturb_prob:
#             neighbors = data_np[indices[i]]
#             pca = PCA(n_components=min(D, k))
#             pca.fit(neighbors)
#             components = pca.components_           # shape: (n_components, D)
#             variances = pca.explained_variance_    # shape: (n_components,)

#             # ノイズベクトル（各主成分方向に沿った合成）
#             noise = np.zeros(D)
#             for j in range(len(components)):
#                     noise += np.random.randn() * np.sqrt(variances[j]) * components[j]

#             # ノイズの方向はそのまま、長さをスケールする
#             if np.linalg.norm(noise) > 0:
#                 noise = noise / np.linalg.norm(noise)

#             # 局所の最大主成分の標準偏差に比例したスケール
#             max_std = np.sqrt(variances[0])  # 最大分散方向
#             scaled_noise = alpha * max_std * noise
#             perturbed_data[i] += scaled_noise
        
#         else:
#             pass

#     return torch.tensor(perturbed_data, dtype=torch.float32).to(device)


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



def compute_avg_knn_distance(features: torch.Tensor, k: int = 10) -> float:
    """
    features: (N, D) Tensor
    k:        近傍数
    戻り値:   全サンプルの k-NN 距離平均
    """
    # 距離行列 (N, N)
    dist = torch.cdist(features, features)
    # 自己距離を除外
    N = features.size(0)
    dist += torch.eye(N, device=features.device) * 1e6
    # k-NN indices
    idx = dist.topk(k=k, largest=False).indices    # (N, k)
    # 距離を gather して平均
    knn_dists = dist.gather(1, idx)               # (N, k)
    return knn_dists.mean().item()

def save_distance_log(distance_log: list, filename: str):
    """
    distance_log: [(epoch, avg_dist), ...]
    filename:     保存先 pickle ファイル
    """
    with open(filename, "wb") as f:
        pickle.dump(distance_log, f)