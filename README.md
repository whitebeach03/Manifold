# Manifold

## セットアップ
`pip install -r requirement.txt`

## 実験
共通引数: 
- `--data_type`: `cifar100`, `cifar10`, `stl10`
- `--model_type`: `wide_resnet_28_10`, `resnet18`, `resnet101`
- `--epochs`: 学習エポック数

### 1. 比較手法の学習 (`train.py`)
```bash
python train.py --data_type [cifar100/cifar10/stl10] --model_type [model] --augment [Default/Mixup/CutMix/Manifold-Mixup/ResizeMix/SaliencyMix/SK-Mixup]
```
### 2. 二段階学習 (`train_phase1.py` → `train_phase2.py`)
#### Phase 1
```bash
python train_phase1.py --data_type [cifar100/cifar10/stl10] --model_type [model] --epochs [num] --augment [Method]
```
#### Phase 2
```bash
python train_phase2.py --data_type [cifar100/cifar10/stl10] --model_type [model] --epochs [num]
```
### 3. 評価
- クリーン精度評価:
```bash
python test_acc.py --data_type [cifar100/cifar10/stl10] --model_type [model] --epochs [num] --augment [Method]
```
- クリーン精度評価:
```bash
python test_acc_c.py
```
# - 比較手法の学習: `python train.py --data_type [cifar100/cifar10/stl10]`
# - 二段階学習: `python train_phase1.py` → `python train_phase2.py`
# - 評価: `python test_acc.py` (クリーン精度評価) / `python test_acc_c.py` (頑健性評価)
