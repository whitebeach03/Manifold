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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, x, labels, device, aug_ok=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        if aug_ok:
            features = out
            # augmented_data = manifold_perturbation(features, device)
            # augmented_data = local_pca_perturbation(features, device)
            # augmented_data = class_pca_perturbation(features, labels, device)
            augmented_data = manifold_perturbation_random(features, device)
            out = self.linear(augmented_data)
        else:
            out = self.linear(out)
        return out



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
    
# def manifold_perturbation(data, device, k=10, noise_scale=0.1):
#     if isinstance(data, torch.Tensor):
#         data_np = data.cpu().detach().numpy()  # PyTorch → NumPy
#     else:
#         data_np = data
#     N, D = data_np.shape
#     augmented_data = np.copy(data_np)
#     k = min(k, N-1)
#     # 最近傍探索
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data_np)
#     distances, indices = nbrs.kneighbors(data_np)
#     for i in range(N):
#         # 近傍点を取得
#         neighbors = data_np[indices[i]]
#         # PCAで局所的な主成分を取得
#         pca = PCA(n_components=min(D, k))
#         pca.fit(neighbors)
#         principal_components = pca.components_  # 主成分方向
#         # 主成分に沿ったランダムノイズを追加
#         noise = np.random.normal(scale=noise_scale, size=(k,))
#         perturbation = np.dot(principal_components.T, noise)
#         # データ点を摂動
#         augmented_data[i] += perturbation
#     # クリッピング（0-1の範囲を維持）
#     augmented_data = np.clip(augmented_data, 0, 1)
#     return torch.tensor(augmented_data, dtype=torch.float32).to(device)


def manifold_perturbation(features, device, epsilon=0.05):
    """
    微小な摂動を特徴空間に加える関数。
    """
    perturbation = torch.randn_like(features, device=device) * epsilon
    perturbed_features = features + perturbation
    return perturbed_features

def manifold_perturbation_random(features, device, epsilon=0.05):
    """
    微小な摂動を特徴空間に加える関数。
    50%の確率で摂動を加える。
    """
    if random.random() < 0.9:
        # ノイズを加える場合
        perturbation = torch.randn_like(features, device=device) * epsilon
        return features + perturbation
    else:
        # そのまま返す
        return features

def local_pca_perturbation(data, device, k=5, noise_scale=0.1):
    """
    局所PCAに基づく摂動をデータに適用する関数
    :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
    :param k: 近傍数
    :param noise_scale: 摂動のスケール
    :return: 摂動後のデータ
    """
    data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    N, D = data_np.shape
    
    # k近傍探索
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
    _, indices = nbrs.kneighbors(data_np)
    
    perturbed_data = np.copy(data_np)
    
    for i in range(N):
        neighbors = data_np[indices[i]]  # 近傍点取得
        
        # PCAを実行
        pca = PCA(n_components=min(D, k))
        pca.fit(neighbors)
        principal_components = pca.components_  # 主成分
        variances = pca.explained_variance_  # 分散（固有値）
        
        # 主成分に沿った摂動を加える
        noise = np.zeros(D)
        for j in range(len(principal_components)):
            noise += np.random.randn() * np.sqrt(variances[j]) * principal_components[j]
        
        # 摂動をデータに加える
        perturbed_data[i] += noise_scale * noise
    
    return torch.tensor(perturbed_data, dtype=torch.float32).to(device)

def class_pca_perturbation(data, labels, device, noise_scale=0.1):
    """
    同じクラスのサンプルを用いた局所PCAに基づく摂動をデータに適用する関数
    :param data: (N, D) 次元のテンソル (N: サンプル数, D: 特徴次元)
    :param labels: (N,) 次元のクラスラベルのテンソル
    :param noise_scale: 摂動のスケール
    :return: 摂動後のデータ
    """
    data_np = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels_np = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels
    N, D = data_np.shape
    
    perturbed_data = np.copy(data_np)
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        class_indices = np.where(labels_np == label)[0]
        class_data = data_np[class_indices]
        
        if len(class_data) < 2:
            continue  # 十分なサンプルがない場合はスキップ
        
        # PCAを実行
        pca = PCA(n_components=min(D, len(class_data)))
        pca.fit(class_data)
        principal_components = pca.components_  # 主成分
        variances = pca.explained_variance_  # 分散（固有値）
        
        for i in class_indices:
            # 主成分に沿った摂動を加える
            noise = np.zeros(D)
            for j in range(len(principal_components)):
                noise += np.random.randn() * np.sqrt(variances[j]) * principal_components[j]
            
            # 摂動をデータに加える
            perturbed_data[i] += noise_scale * noise
    
    return torch.tensor(perturbed_data, dtype=torch.float32).to(device)
