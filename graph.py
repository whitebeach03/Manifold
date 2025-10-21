import pickle
import matplotlib.pyplot as plt

file_path = './history/wide_resnet_28_10/Mixup-FOMA2/cifar100_400_3.pickle'

# === 1. pickleファイルの読み込み ===
# 例: "accuracy_per_epoch.pkl" に保存されているとする
with open(file_path, 'rb') as f:
    accuracy_list = pickle.load(f)

val_acc = accuracy_list['val_accuracy']
epochs = range(1, len(val_acc) + 1)

plt.plot(epochs, val_acc, linestyle='solid', linewidth=0.8, label='Mixup-FOMA2')

# === 2. データの形式確認 ===
# accuracy_list がリストや numpy 配列を想定
# 各エポックごとの正解率 [0.85, 0.87, 0.89, ...] のような形式
# if not isinstance(accuracy_list, (list, tuple)):
#     raise ValueError("正解率データがリスト形式ではありません。")

# === 3. グラフ描画 ===
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# epochs = list(range(1, len(accuracy_list) + 1))

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, accuracy_list, marker='o', linewidth=2)
# plt.title('Model Accuracy per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
