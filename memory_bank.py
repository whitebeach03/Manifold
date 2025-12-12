import torch

class FeatureMemoryBank:
    def __init__(self, feature_dim: int, memory_size: int, num_classes: int = 100):
        """
        CC-FOMA用 Feature Memory Bank
        
        Args:
            feature_dim (int): 特徴量の次元数 D
            memory_size (int): 保存するサンプルの総数 M (例: 4096, 8192)
            num_classes (int): クラス数 (One-hot変換等のために保持)
        """
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.num_classes = num_classes
        self.ptr = 0
        self.size = 0
        
        # CPU/GPUはupdate時に動的に判定するため、初期化時はdevice指定しない
        # あるいは register_buffer のように扱う
        self.features = None
        self.labels = None
        self.device = None

    def _init_memory(self, device):
        """最初のupdate時にメモリを確保する"""
        self.features = torch.zeros(self.memory_size, self.feature_dim, device=device)
        self.labels = torch.zeros(self.memory_size, dtype=torch.long, device=device)
        self.device = device

    def update(self, batch_features: torch.Tensor, batch_labels: torch.Tensor):
        """
        現在のバッチでメモリバンクを更新する
        
        Args:
            batch_features (Tensor): (B, D)
            batch_labels (Tensor): (B,) or (B, C) one-hot
        """
        # One-hotならインデックスに戻す
        if batch_labels.ndim > 1:
            batch_labels = batch_labels.argmax(dim=1)
            
        batch_features = batch_features.detach() # 勾配は切る
        batch_labels = batch_labels.detach()
        
        batch_size = batch_features.shape[0]
        
        # 初回初期化
        if self.features is None:
            self._init_memory(batch_features.device)
            
        # リングバッファのポインタ計算
        assert batch_size <= self.memory_size, "Batch size larger than memory size"
        
        if self.ptr + batch_size <= self.memory_size:
            self.features[self.ptr : self.ptr + batch_size] = batch_features
            self.labels[self.ptr : self.ptr + batch_size] = batch_labels
            self.ptr = (self.ptr + batch_size) % self.memory_size
        else:
            tail = self.memory_size - self.ptr
            head = batch_size - tail
            self.features[self.ptr :] = batch_features[:tail]
            self.labels[self.ptr :] = batch_labels[:tail]
            self.features[:head] = batch_features[tail:]
            self.labels[:head] = batch_labels[tail:]
            self.ptr = head
            
        self.size = min(self.size + batch_size, self.memory_size)

    def get_memory(self):
        """現在保存されている有効な特徴量とラベルを返す"""
        if self.features is None or self.size == 0:
            return None, None
            
        # まだメモリが満杯でない場合は有効な部分だけ返す
        if self.size < self.memory_size:
            return self.features[:self.size], self.labels[:self.size]
        else:
            return self.features, self.labels