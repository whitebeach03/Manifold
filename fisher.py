import torch
import torchvision
import torchvision.transforms as transforms
from src.utils import *
from tqdm import tqdm
from collections import defaultdict
from src.models.resnet import ResNet18
from src.models.wide_resnet import Wide_ResNet
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split

def main():
    iteration  = 1
    epochs     = 400
    data_type  = "cifar100"
    model_type = "wide_resnet_28_10"
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    augmentations = [
        "Default",
        "Mixup",
        # "Manifold-Mixup",
        "Local-FOMA",
    ]

    for augment in augmentations:
        print(f"============{augment}============")
        fisher_list = []
        snn_list    = []
        for i in range(iteration):

            model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_{i}.pth"

            if data_type == "stl10":
                num_classes = 10
                batch_size  = 64
            elif data_type == "cifar100":
                num_classes = 100
                batch_size  = 128
            elif data_type == "cifar10":
                num_classes = 10
                batch_size  = 128
            
            if model_type == "resnet18":
                model = ResNet18().to(device)
            elif model_type == "wide_resnet_28_10":
                model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
            
            if data_type == "stl10":
                transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                test_dataset  = STL10(root="./data", split="train", download=True, transform=transform)
            elif data_type == "cifar100":
                transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                test_dataset  = CIFAR100(root="./data", train=False, transform=transform, download=True)
            elif data_type == "cifar10":
                transform     = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                test_dataset  = CIFAR10(root="./data", train=False, transform=transform, download=True)
            
            test_loader  = DataLoader(dataset=test_dataset,  batch_size=256, shuffle=False)

            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            model.eval()

            features_list = []
            labels_list = []
            with torch.no_grad():
                for images, labels in tqdm(test_loader, leave=False):
                    images = images.to(device)

                    features = model.extract_features(images)
                    features_list.append(features.cpu())
                    labels_list.append(labels)
            features = torch.cat(features_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            fisher_ratio = compute_fisher_discriminant_ratio_normed(features, labels, normalize="standard")
            snn          = soft_nearest_neighbor_loss(features, labels)
            fisher_list.append(fisher_ratio)
            snn_list.append(snn)

        fisher_ratio_avg = cal_average(fisher_list)
        snn_avg = cal_average(snn_list)
        print(f"Fisher判別比: {fisher_ratio_avg:.4f}")
        print(f"Soft Nearest Neighbor Loss: {snn_avg:.4f}")

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           
    avg = sum_num / len(num)
    return avg

def soft_nearest_neighbor_loss(features, labels, temperature=0.1):
    """
    features: [N, D] テンソル（N: サンプル数, D: 特徴次元）
    labels: [N] ラベル
    temperature: 距離のスケーリング係数（小さいほど鋭くなる）
    """
    N = features.size(0)
    dist_matrix = torch.cdist(features, features, p=2) ** 2  # 距離の2乗

    mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # 同クラス判定マスク（NxN）
    mask.fill_diagonal_(False)  # 自分自身は除く

    logits = -dist_matrix / temperature
    exp_logits = torch.exp(logits)

    num = (exp_logits * mask).sum(dim=1)  # 分子：同クラスのみ
    denom = exp_logits.sum(dim=1)         # 分母：全体

    loss = -torch.log((num + 1e-8) / (denom + 1e-8))
    return loss.mean()

def compute_fisher_discriminant_ratio(features: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fisher判別比（クラス間分散 / クラス内分散）を計算する

    Parameters:
        features (torch.Tensor): NxDのテンソル（N: サンプル数, D: 特徴次元）
        labels (torch.Tensor): 長さNの整数ラベル（各サンプルのクラス）

    Returns:
        float: Fisher判別比（trace(S_B) / trace(S_W)）
    """

    device = features.device
    labels = labels.cpu().numpy()
    features = features.cpu()
    class_features = defaultdict(list)

    # クラスごとに特徴をグループ化
    for f, l in zip(features, labels):
        class_features[int(l)].append(f)

    all_features = features
    global_mean = all_features.mean(dim=0)

    # 初期化
    num_features = features.shape[1]
    Sw = torch.zeros((num_features, num_features))
    Sb = torch.zeros((num_features, num_features))

    for c, feats in class_features.items():
        feats_tensor = torch.stack(feats)
        class_mean = feats_tensor.mean(dim=0)
        n_c = feats_tensor.size(0)

        # クラス内分散（Sw）
        diff = feats_tensor - class_mean
        Sw += diff.T @ diff

        # クラス間分散（Sb）
        mean_diff = (class_mean - global_mean).unsqueeze(1)
        Sb += n_c * (mean_diff @ mean_diff.T)
    
    ##################################
    N = all_features.size(0)
    # ← ここで総和を N で割り「平均散布行列」に
    Sw /= N
    Sb /= N
    ##################################

    # trace-based Fisher比（スカラー値）
    fisher_ratio = torch.trace(Sb) / (torch.trace(Sw) + 1e-8)  # 0除算防止

    print("クラス内分散: ", torch.trace(Sw))
    print("クラス間分散: ", torch.trace(Sb))

    return fisher_ratio.item()


import torch
import torch.nn.functional as F
from collections import defaultdict

def compute_fisher_discriminant_ratio_normed(
    features: torch.Tensor,
    labels: torch.Tensor,
    normalize: str = "unit",  # "unit" or "standard"
) -> float:
    """
    Fisher判別比を、特徴のスケールを揃えた上で計算する版。

    normalize:
      - "unit"     : 各サンプルをL2ノルム=1に
      - "standard" : 各次元を平均0・分散1に
    """

    # 1) CPU上でnumpyに落としたりしない。全てTensorで扱う
    device = features.device
    labels = labels.to(device)

    # 2) 前処理：スケールを揃える
    if normalize == "unit":
        # 各行（サンプル）のノルムを1に
        features = F.normalize(features, p=2, dim=1)
    elif normalize == "standard":
        # 各列（次元）を平均0・分散1に
        mu    = features.mean(dim=0, keepdim=True)
        sigma = features.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        features = (features - mu) / sigma

    # 3) 以下は従来どおり Sw, Sb を「平均散布行列」で計算
    N, D = features.shape
    global_mean = features.mean(dim=0)

    Sw = torch.zeros((D, D), device=device)
    Sb = torch.zeros((D, D), device=device)
    labels_np = labels.cpu().numpy()
    class_feats = defaultdict(list)
    for x, l in zip(features, labels_np):
        class_feats[int(l)].append(x)

    for feats in class_feats.values():
        feats = torch.stack(feats, dim=0)
        m_c   = feats.mean(dim=0)
        diff  = feats - m_c
        Sw   += diff.t() @ diff
        md    = (m_c - global_mean).unsqueeze(1)
        Sb   += feats.size(0) * (md @ md.t())

    # 平均散布行列に
    Sw /= N
    Sb /= N

    # 結果表示
    print(f"[normalize={normalize}] クラス内分散 = {torch.trace(Sw):.4f},  クラス間分散 = {torch.trace(Sb):.4f}")

    # Fisher比
    return (torch.trace(Sb) / (torch.trace(Sw) + 1e-8)).item()


if __name__ == "__main__":
    main()