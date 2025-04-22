import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = conv3x3(3,64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, labels, device, augment, k=10, aug_ok=False, mixup_hidden=False):
        if mixup_hidden == False:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, out.size()[2])
            out = out.view(out.size(0), -1)
            if aug_ok:
                features = out
                if augment == "normal":
                    augmented_data = features
                elif augment == "perturb":
                    augmented_data = manifold_perturbation(features, device)
                elif augment == "PCA":
                    augmented_data = local_pca_perturbation(features, device, k)
                elif augment == "PCA-2012":
                    augmented_data = pca_directional_perturbation_local(features, device, k)
                else:
                    augmented_data = local_pca_perturbation(features, device, k)

                out = self.linear(augmented_data)
            else:
                out = self.linear(out)
            return out

        else:
            mixup_alpha = 0.1
            # layer_mix = random.randint(0,5)
            layer_mix = 10
            out = x
            
            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)
            
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

            if layer_mix == 10:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            out = self.linear(out)
            
            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data_hidden(out, labels, mixup_alpha)

            return out, y_a, y_b, lam

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        return out  # ← 特徴ベクトル


def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,1,96,96)))
    print(y.size())

def manifold_perturbation(features, device, epsilon=0.05):
    """
    微小な摂動を特徴空間に加える関数。
    """
    perturbation = torch.randn_like(features, device=device) * epsilon
    perturbed_features = features + perturbation
    return perturbed_features

def manifold_perturbation_random(features, device, random_rate, epsilon=0.05):
    """
    微小な摂動を特徴空間に加える関数。
    50%の確率で摂動を加える。
    """
    if random.random() < random_rate:
        # ノイズを加える場合
        perturbation = torch.randn_like(features, device=device) * epsilon
        return features + perturbation
    else:
        # そのまま返す
        return features

# def local_pca_perturbation(data, labels, device, k=10, alpha=1.0):
#     """
#     局所PCAに基づく摂動をデータに加える（同クラスかつ近傍の散らばり内に収める）

#     :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
#     :param labels: (N,) 次元のラベルテンソル
#     :param device: 使用するデバイス（cuda or cpu）
#     :param k: k近傍の数
#     :param alpha: 摂動の強さ（最大主成分の標準偏差に対する割合）
#     :return: 摂動後のテンソル（同shape）
#     """
#     use_variance_scaling = True
#     data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
#     labels_np = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels
#     N, D = data_np.shape

#     perturbed_data = np.copy(data_np)

#     for i in range(N):
#         # 同じクラスのインデックスを取得
#         same_class_mask = labels_np == labels_np[i]
#         same_class_data = data_np[same_class_mask]

#         if len(same_class_data) <= 1:
#             continue  # 自分しかいない場合はスキップ

#         # k近傍探索（同クラス内で）
#         k_eff = min(k, len(same_class_data))
#         nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm='ball_tree').fit(same_class_data)
#         _, indices = nbrs.kneighbors([data_np[i]])
#         neighbors = same_class_data[indices[0]]

#         # PCA実行
#         pca = PCA(n_components=min(D, k_eff))
#         pca.fit(neighbors)
#         components = pca.components_           # shape: (n_components, D)
#         variances = pca.explained_variance_    # shape: (n_components,)

#         # ノイズベクトル作成
#         noise = np.zeros(D)
#         for j in range(len(components)):
#             if use_variance_scaling:
#                 noise += np.random.randn() * np.sqrt(variances[j]) * components[j]
#             else:
#                 noise += np.random.randn() * components[j]

#         # 正規化＆スケーリング
#         if np.linalg.norm(noise) > 0:
#             noise = noise / np.linalg.norm(noise)
#         max_std = np.sqrt(variances[0])
#         scaled_noise = alpha * max_std * noise

#         # 摂動適用
#         perturbed_data[i] += scaled_noise

#     return torch.tensor(perturbed_data, dtype=torch.float32).to(device)

def local_pca_perturbation(data, device, k=10, alpha=1.0):
    """
    局所PCAに基づく摂動をデータに加える（近傍の散らばり内に収める）
    :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
    :param device: 使用するデバイス（cuda or cpu）
    :param k: k近傍の数
    :param alpha: 摂動の強さ（最大主成分の標準偏差に対する割合）
    :return: 摂動後のテンソル（同shape）
    """
    use_variance_scaling = True
    data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    N, D = data_np.shape
    if N < k:
        k = N

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
    _, indices = nbrs.kneighbors(data_np)

    perturbed_data = np.copy(data_np)

    for i in range(N):
        neighbors = data_np[indices[i]]
        pca = PCA(n_components=min(D, k))
        pca.fit(neighbors)
        components = pca.components_           # shape: (n_components, D)
        variances = pca.explained_variance_    # shape: (n_components,)

        # ノイズベクトル（各主成分方向に沿った合成）
        noise = np.zeros(D)
        for j in range(len(components)):
            if use_variance_scaling:
                noise += np.random.randn() * np.sqrt(variances[j]) * components[j]
            else:
                noise += np.random.randn() * components[j]

        # ノイズの方向はそのまま、長さをスケールする
        if np.linalg.norm(noise) > 0:
            noise = noise / np.linalg.norm(noise)

        # 局所の最大主成分の標準偏差に比例したスケール
        max_std = np.sqrt(variances[0])  # 最大分散方向
        scaled_noise = alpha * max_std * noise

        perturbed_data[i] += scaled_noise

    return torch.tensor(perturbed_data, dtype=torch.float32).to(device)

def pca_directional_perturbation_local(data, device, k=10, alpha=1.0):
    """
    各特徴ベクトルに対して、k近傍のPCA主成分に基づき摂動を加える（Local PCA）
    """
    data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    N, D = data_np.shape
    k = min(k, N)

    # k-NN を構築
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
    _, indices = nbrs.kneighbors(data_np)

    perturbed_data = []

    for i in range(N):
        xi = data_np[i]
        neighbors = data_np[indices[i]]  # k近傍を取得

        pca = PCA(n_components=min(k, D))
        pca.fit(neighbors)

        components = pca.components_        # shape: (k, D)
        variances = pca.explained_variance_ # shape: (k,)

        noise = np.zeros(D)
        for j in range(len(components)):
            a_j = np.random.randn()
            noise += a_j * variances[j] * components[j]  # 分散スケーリング

        xi_aug = xi + alpha * noise
        perturbed_data.append(xi_aug)

    return torch.tensor(np.array(perturbed_data), dtype=torch.float32).to(device)


def mixup_data_hidden(x, y, alpha, use_cuda=True):
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