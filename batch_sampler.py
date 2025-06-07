
import random
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from src.models.wide_resnet import Wide_ResNet
from collections import defaultdict
import random
import numpy as np
from torch.utils.data import Sampler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import math

class HybridFOMABatchSampler(Sampler):
    """
    Hybrid sampler for FOMA_knn: each batch consists of
      1) Local-KNN frame: α * B samples from each sample's static k-NN
      2) Class-Stratified/Diverse frame: (1 - α) * B samples from diverse classes

    Arguments:
        feature_matrix (np.ndarray): shape (n_samples, feature_dim), features from a fixed pretrained model.
        labels (list[int]): length n_samples, label for each sample in the same order as feature_matrix.
        batch_size (int): desired batch size B.
        alpha (float): ratio of local‐KNN frame in a batch (0.0 <= α <= 1.0).
        k_neighbors (int or None): number of neighbors to compute per sample; if None, defaults to ceil(alpha * B).
        drop_last (bool): if True, drop the last incomplete batch; else pad it to full size.
    """
    def __init__(self, feature_matrix, labels, batch_size, alpha=0.6, k_neighbors=None, drop_last=False):
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.batch_size = batch_size
        self.alpha = alpha
        self.drop_last = drop_last

        self.n_samples = feature_matrix.shape[0]
        # バッチ数を計算 (drop_last=True なら floor, False なら ceil)
        if self.drop_last:
            self.num_batches = self.n_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.n_samples / self.batch_size)

        # k_neighbors を指定しない場合は α*B-1 (local_neighbors に self を含めるため)
        if k_neighbors is None:
            needed = max(1, int(self.alpha * self.batch_size) - 1)
            self.k_neighbors = needed + 1  # +1 で self を含める
        else:
            self.k_neighbors = k_neighbors

        # static k-NN を構築（一度だけ）
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto', metric='euclidean')
        nbrs.fit(self.feature_matrix)
        distances, neighbors = nbrs.kneighbors(self.feature_matrix)
        # 各サンプル i の近傍リストを作成
        # 「[i] + (i の上位 (αB - 1) の近傍)」を local_neighbors[i] として保持
        self.local_neighbors = []
        num_local = max(1, int(self.alpha * self.batch_size))  # 少なくとも 1 枚 (center 自身)
        for i in range(self.n_samples):
            neighs = neighbors[i].tolist()
            # neighs[0] はおそらく i 自身なので取り除く
            if neighs[0] == i:
                neighs = neighs[1:]
            chosen = neighs[: (num_local - 1)]  # (αB - 1) 枚を選択
            self.local_neighbors.append([i] + chosen)  # center + chosen

        # ラベルごとのインデックスを保持 -> Class-Stratified 用に使う
        self.class_to_indices = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.class_to_indices[lbl].append(idx)
        self.all_classes = list(self.class_to_indices.keys())

    def __iter__(self):
        """
        実際にバッチごとのインデックスを yield する。
        返すバッチ数は self.num_batches に合わせる。
        """
        # ランダムに「バッチ数の分だけ」センターを選ぶ
        # n_samples >= num_batches 前提。もし n_samples が小さい場合は重複を許す。
        if self.n_samples >= self.num_batches:
            centers = random.sample(range(self.n_samples), self.num_batches)
        else:
            # サンプル数 < バッチ数 の場合は、サンプルを重複して選ぶ
            centers = [random.randrange(self.n_samples) for _ in range(self.num_batches)]

        for center in centers:
            # (1) Local-KNN 枠を構築
            local_batch = self.local_neighbors[center].copy()
            if len(local_batch) < int(self.alpha * self.batch_size):
                # もし「近傍リストが足りない」場合はランダムで補填
                needed_local = int(self.alpha * self.batch_size) - len(local_batch)
                pool = [i for i in range(self.n_samples) if i not in local_batch]
                if len(pool) >= needed_local:
                    local_batch += random.sample(pool, needed_local)
                else:
                    local_batch += pool
            num_local = len(local_batch)

            # (2) Class-Stratified / Diverse 枠を構築
            num_diverse = self.batch_size - num_local
            diverse_batch = []
            random.shuffle(self.all_classes)
            cls_ptr = 0
            while len(diverse_batch) < num_diverse and cls_ptr < len(self.all_classes):
                cls = self.all_classes[cls_ptr]
                cls_ptr += 1
                candidates = self.class_to_indices[cls]
                # local_batch と diverse_batch に含まれないものを候補にする
                pool = [c for c in candidates if c not in local_batch and c not in diverse_batch]
                if not pool:
                    continue
                # 1 枚ランダムに選ぶ
                pick = random.choice(pool)
                diverse_batch.append(pick)
            # それでもバッチが埋まらない場合は全体からランダムに補填
            if len(diverse_batch) < num_diverse:
                needed = num_diverse - len(diverse_batch)
                pool_all = [
                    i for i in range(self.n_samples)
                    if i not in local_batch and i not in diverse_batch
                ]
                if len(pool_all) >= needed:
                    diverse_batch += random.sample(pool_all, needed)
                else:
                    diverse_batch += pool_all

            # (3) 両方を合わせてシャッフル
            batch_indices = local_batch + diverse_batch
            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[: self.batch_size]
            elif len(batch_indices) < self.batch_size and not self.drop_last:
                needed = self.batch_size - len(batch_indices)
                pool_pad = [
                    i for i in range(self.n_samples)
                    if i not in batch_indices
                ]
                if len(pool_pad) >= needed:
                    batch_indices += random.sample(pool_pad, needed)
                else:
                    batch_indices += pool_pad
            random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches


def extract_wrn_features(model: Wide_ResNet, dataset, device, batch_size=256, num_workers=4):
    """
    Wide-ResNet の extract_features メソッドを使い、
    `dataset` の全サンプルに対して 512-D 特徴を算出し、NumPy 配列で返します。

    Arguments:
        model       : Wide_ResNet のインスタンス (eval モード)
        dataset     : torch.utils.data.Dataset (train のサブセットなど)
        device      : torch.device('cuda') or 'cpu'
        batch_size  : DataLoader のバッチサイズ (特徴抽出時に使う)
        num_workers : DataLoader の num_workers

    Returns:
        features_np : shape=(len(dataset), 512) の NumPy 配列
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_feats = []

    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extract WRN Features"):
            images = images.to(device)
            feats = model.extract_features(images)  # [B, 512]
            all_feats.append(feats.cpu().numpy())

    features_np = np.concatenate(all_feats, axis=0)  # (n_samples, 512)
    return features_np


class FeatureKNNBatchSampler(Sampler):
    """
    事前計算した特徴行列 (n_samples x 512) を使って、
    バッチサイズ (k) 個の「k-NN インデックスリスト」を生成する Sampler。
    """

    def __init__(self, feature_matrix: np.ndarray, batch_size: int):
        """
        feature_matrix : (n_samples, 512) の NumPy 配列
        batch_size     : 1 バッチあたりのサンプル数 ( k-近傍数)
        """
        super().__init__(None)
        self.feature_matrix = feature_matrix
        self.batch_size = batch_size
        self.N = feature_matrix.shape[0]

        # Scikit-learn の NearestNeighbors を使ってインデックスを構築
        self.nn_model = NearestNeighbors(
            n_neighbors=self.batch_size,
            algorithm='auto',
            metric='euclidean'
        )
        self.nn_model.fit(self.feature_matrix)

    def __iter__(self):
        # エポックあたり N//batch_size 個のバッチを生成
        num_batches = self.N // self.batch_size
        all_indices = np.arange(self.N)
        np.random.shuffle(all_indices)

        for i in range(num_batches):
            seed_idx = int(all_indices[i])
            # シードの k-NN インデックス (自身含む) を取得
            _, neighbors = self.nn_model.kneighbors(
                self.feature_matrix[seed_idx].reshape(1, -1),
                n_neighbors=self.batch_size,
                return_distance=True
            )
            batch_indices = neighbors[0].tolist()
            yield batch_indices

    def __len__(self):
        return self.N // self.batch_size
