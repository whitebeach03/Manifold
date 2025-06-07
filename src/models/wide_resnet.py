import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from foma import foma
import sys

class FOMALayer(nn.Module):
    def __init__(self, feature_dim, num_classes, alpha=1.0, rho=0.9, small_singular=True):
        super(FOMALayer, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.rho = rho
        self.small_singular = small_singular

    def forward(self, X, Y, apply_foma=True):
        """
        X: 中間特徴 [B, D] （flattenされた中間表現）
        Y: ラベル [B] または [B, num_classes]
        apply_foma: FOMA適用フラグ
        """
        if not apply_foma:
            return X, Y

        B = X.shape[0]

        # one-hot変換
        if Y.ndim == 1:
            Y_onehot = F.one_hot(Y, num_classes=self.num_classes).float()
        else:
            Y_onehot = Y.float()

        # Z = [X | Y]
        Z = torch.cat([X, Y_onehot], dim=1)

        # SVD
        U, s, Vt = torch.linalg.svd(Z, full_matrices=False)

        # ラムダ決定
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().to(X.device)
        cumperc = torch.cumsum(s, dim=0) / torch.sum(s)
        condition = cumperc > self.rho if self.small_singular else cumperc < self.rho
        lam_mult = torch.where(condition, lam, torch.tensor(1.0, device=s.device))
        s_scaled = s * lam_mult

        # 再構成
        Z_scaled = (U @ torch.diag(s_scaled) @ Vt)
        X_scaled = Z_scaled[:, :X.shape[1]]
        Y_scaled = Z_scaled[:, X.shape[1]:]

        # ラベルをソフトに正規化
        Y_scaled = torch.clamp(Y_scaled, min=0)
        sum_per_sample = Y_scaled.sum(dim=1, keepdim=True)
        normalized_labels = torch.where(
            sum_per_sample == 0,
            torch.ones_like(Y_scaled) / self.num_classes,
            Y_scaled / sum_per_sample
        )

        return X_scaled, normalized_labels

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.foma_layer = FOMALayer(feature_dim=512, num_classes=num_classes)

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, labels, device, augment, k=10, aug_ok=False, mixup_hidden=False, num_classes=100):
        if mixup_hidden == False:
            if augment == "FOMA_latent_random":
                if aug_ok:
                    layer_foma = random.randint(0, 4)
                    out = x

                    if layer_foma == 0:
                        out, y_soft = foma(out, labels, num_classes, alpha=1.0, rho=0.9)

                    # conv1 → layer1 → (maybe foma) → layer2 → (maybe foma) → layer3 → ... 
                    out = self.conv1(out)
                    out = self.layer1(out)   # ← You must pass through layer1 here.
                    if layer_foma == 1:
                        out, y_soft = foma(out, labels, num_classes, alpha=1.0, rho=0.9)

                    out = self.layer2(out)
                    if layer_foma == 2:
                        out, y_soft = foma(out, labels, num_classes, alpha=1.0, rho=0.9)

                    out = self.layer3(out)
                    if layer_foma == 3:
                        out, y_soft = foma(out, labels, num_classes, alpha=1.0, rho=0.9)

                    out = F.relu(self.bn1(out))
                    out = F.avg_pool2d(out, 8)
                    out = out.view(out.size(0), -1)
                    if layer_foma == 4:
                        out, y_soft = foma(out, labels, num_classes, alpha=1.0, rho=0.9)

                    out = self.linear(out)
                    return out, y_soft
                else:
                    out = self.conv1(x)
                    out = self.layer1(out)
                    out = self.layer2(out)
                    out = self.layer3(out)
                    out = F.relu(self.bn1(out))
                    out = F.avg_pool2d(out, 8)
                    out = out.view(out.size(0), -1)
                    out = self.linear(out)
                    return out
                
            else:
                out = self.conv1(x)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = F.relu(self.bn1(out))
                out = F.avg_pool2d(out, 8)
                out = out.view(out.size(0), -1)
                if aug_ok:
                    if augment == "FOMA_latent" or augment == "FOMA_curriculum":
                        out, y_soft = foma(out, labels, num_classes, alpha=1.0, rho=0.9)
                        out = self.linear(out)
                        return out, y_soft

                    else:
                        out = self.linear(out)
                        return out
                else:
                    out = self.linear(out)
                    return out
            
        else:
            mixup_alpha = 0.1
            layer_mix = random.randint(1,5)
            out = x
            
            # if layer_mix == 0:
            #     out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
    
            if layer_mix == 1:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            out = self.layer2(out)
    
            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            out = self.layer3(out)
            
            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            out = self.layer4(out)
            
            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            # out = F.avg_pool2d(out, 4)
            out = F.avg_pool2d(out, out.size()[2])
            out = out.view(out.size(0), -1)

            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            out = self.linear(out)
            return out, y_a, y_b, lam
    
    def extract_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out


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

# def local_pca_perturbation(data, device, k=10, alpha=1.0, perturb_prob=1.0, variance_threshold=0.9):
#     """
#     局所PCAに基づく摂動をデータに加える（高分散方向のみに制限）
#     :param data: (N, D) 次元のテンソル
#     :param device: cuda or cpu
#     :param k: k近傍数
#     :param alpha: ノイズスケール（最大主成分に対する係数）
#     :param perturb_prob: ノイズ付加の確率
#     :param variance_threshold: 寄与率の累積で使用するカットオフ（例：0.9）
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

#             components = pca.components_
#             variances = pca.explained_variance_
#             explained_ratio = pca.explained_variance_ratio_

#             # 累積寄与率に基づいて主成分を選ぶ
#             cumulative = np.cumsum(explained_ratio)
#             valid_indices = np.where(cumulative <= variance_threshold)[0]
#             if len(valid_indices) == 0:
#                 valid_indices = [0]  # 少なくとも1成分は使う

#             # ノイズベクトル
#             noise = np.zeros(D)
#             for j in valid_indices:
#                 noise += np.random.randn() * np.sqrt(variances[j]) * components[j]

#             if np.linalg.norm(noise) > 0:
#                 noise = noise / np.linalg.norm(noise)

#             max_std = np.sqrt(variances[0])
#             scaled_noise = alpha * max_std * noise
#             perturbed_data[i] += scaled_noise

#     return torch.tensor(perturbed_data, dtype=torch.float32).to(device)
