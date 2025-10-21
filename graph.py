# import pickle
# import matplotlib.pyplot as plt

# file_path = './history/wide_resnet_28_10/Mixup-FOMA2/cifar100_400_3.pickle'

# with open(file_path, 'rb') as f:
#     accuracy_list = pickle.load(f)

# val_acc = accuracy_list['val_accuracy']
# epochs = range(1, len(val_acc) + 1)

# plt.plot(epochs, val_acc, linestyle='solid', linewidth=0.8, label='Mixup-FOMA2')

# plt.title('Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()


import pickle
import matplotlib.pyplot as plt

# === 1. pickleファイルの読み込み ===
with open('/home/shirahama/Manifold/history/wide_resnet_28_10/RegMixup/cifar100_360_2.pickle', 'rb') as f:
    data_360 = pickle.load(f)

with open('/home/shirahama/Manifold/history/wide_resnet_28_10/Mixup-FOMA2/cifar100_400_2.pickle', 'rb') as f:
    data_400 = pickle.load(f)

# with open('/home/shirahama/Manifold/history/wide_resnet_28_10/Mixup/cifar100_400_2.pickle', 'rb') as f:
#     mixup = pickle.load(f)

# === 2. データの連結 ===
# 360エポック学習済み → 360個
# 追加40エポック → 40個
# 注意: data_400 の最初の部分が360と重複していないか確認
val_acc = data_360['val_accuracy'] + data_400['val_accuracy']
# val_mixup = mixup['val_accuracy']

# === 3. グラフ描画 ===
epochs = list(range(1, len(val_acc) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_acc, label='Mixup-FOMA', linewidth=2)
# plt.plot(epochs, val_mixup, label='Mixup', linewidth=2)
plt.title('Model Accuracy over 400 Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
