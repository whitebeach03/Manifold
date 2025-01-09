import pickle
import matplotlib.pyplot as plt

def plot_training_history(pickle_file_path):
    # 保存された history ファイルをロード
    with open(pickle_file_path, 'rb') as f:
        history = pickle.load(f)

    # エポック数を取得
    epochs = range(1, len(history['loss']) + 1)

    # プロットを作成
    plt.figure(figsize=(12, 5))

    # Lossのプロット
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracyのプロット
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./result_plot/resnet18/stl10_200.png')
    plt.show()

# 使用例
if __name__ == "__main__":
    pickle_file_path = './history/resnet18/stl10_200.pickle'
    plot_training_history(pickle_file_path)
