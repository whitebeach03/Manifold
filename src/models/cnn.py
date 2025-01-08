import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """表現力の高いCNNモデル"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 畳み込み層1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # 64チャネル
        self.bn1 = nn.BatchNorm2d(64)
        
        # 畳み込み層2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 128チャネル
        self.bn2 = nn.BatchNorm2d(128)
        
        # 畳み込み層3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 256チャネル
        self.bn3 = nn.BatchNorm2d(256)
        
        # 畳み込み層4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # 512チャネル
        self.bn4 = nn.BatchNorm2d(512)

        # プーリング層
        self.pool = nn.MaxPool2d(2, 2)

        # 全結合層に置き換えるためのグローバル平均プーリング
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全結合層
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)  # 10クラス分類

        # ドロップアウト
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 畳み込み層1 + バッチ正規化 + ReLU + プーリング
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 畳み込み層2 + バッチ正規化 + ReLU + プーリング
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 畳み込み層3 + バッチ正規化 + ReLU + プーリング
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 畳み込み層4 + バッチ正規化 + ReLU + プーリング
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # グローバル平均プーリング
        x = self.global_avg_pool(x)  # 出力サイズ: (N, 512, 1, 1)
        x = x.view(-1, 512)          # フラット化 (N, 512)

        # 全結合層1 + ドロップアウト + ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 出力層
        x = self.fc2(x)

        return x