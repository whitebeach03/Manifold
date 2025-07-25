import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from src.methods.pca import local_pca_perturbation, fast_batch_pca_perturbation
# from src.methods.foma import foma
# from src.utils import mixup_data, fast_batch_pca_perturbation

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
        self.bn1     = nn.BatchNorm2d(in_planes)
        self.conv1   = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2     = nn.BatchNorm2d(planes)
        self.conv2   = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out  = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out  = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1  = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1    = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, labels, device, augment, k=10, aug_ok=False, num_classes=100):
        if augment == "Manifold-Mixup":
            if aug_ok:
                mixup_alpha = 2.0
                layer_mix = random.randint(0,4)
                out = x
                
                if layer_mix == 0:
                    out, y_a, y_b, lam = mixup_data(out, labels, mixup_alpha)
                
                out = self.conv1(out)
                out = self.layer1(out)
                if layer_mix == 1:
                    out, y_a, y_b, lam = mixup_data(out, labels, mixup_alpha)

                out = self.layer2(out)
                if layer_mix == 2:
                    out, y_a, y_b, lam = mixup_data(out, labels, mixup_alpha)

                out = self.layer3(out)
                if layer_mix == 3:
                    out, y_a, y_b, lam = mixup_data(out, labels, mixup_alpha)
                
                out = F.relu(self.bn1(out))
                out = F.avg_pool2d(out, 8)
                out = out.view(out.size(0), -1)
                if layer_mix == 4:
                    out, y_a, y_b, lam = mixup_data(out, labels, mixup_alpha)

                out = self.linear(out)
                return out, y_a, y_b, lam
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

        elif augment == "Manifold-SK-Mixup":
            if aug_ok:
                layer_mix = random.randint(0,4)
                out = x
                
                if layer_mix == 0:
                    out, y_a, y_b, lam = sk_mixup_feature(f=out, y=labels, tau_max=1.0, tau_std=0.25, device=device)
                
                out = self.conv1(out)
                out = self.layer1(out)
                if layer_mix == 1:
                    out, y_a, y_b, lam = sk_mixup_feature(f=out, y=labels, tau_max=1.0, tau_std=0.25, device=device)

                out = self.layer2(out)
                if layer_mix == 2:
                    out, y_a, y_b, lam = sk_mixup_feature(f=out, y=labels, tau_max=1.0, tau_std=0.25, device=device)

                out = self.layer3(out)
                if layer_mix == 3:
                    out, y_a, y_b, lam = sk_mixup_feature(f=out, y=labels, tau_max=1.0, tau_std=0.25, device=device)
                
                out = F.relu(self.bn1(out))
                out = F.avg_pool2d(out, 8)
                out = out.view(out.size(0), -1)
                if layer_mix == 4:
                    out, y_a, y_b, lam = sk_mixup_feature(f=out, y=labels, tau_max=1.0, tau_std=0.25, device=device)

                out = self.linear(out)
                return out, y_a, y_b, lam
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
            out = self.linear(out)
            return out
    
    def extract_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out

def count_cnn_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = Wide_ResNet(28, 10, 0.3, num_classes=100)
    trainable_params = count_cnn_parameters(model, only_trainable=True)
    print(f"Trainable parameters: {trainable_params:,}")