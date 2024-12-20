import matplotlib.pyplot as plt
import pickle
import numpy as np

def load_history(file_path):
    """pickleファイルから履歴データを読み込む"""
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    return history

def plot_val_accuracy(history):
    """val_accをプロットする"""
    val_accuracy = history['val_accuracy']
    epochs = range(1, len(val_accuracy) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
    plt.xticks(np.arange(0, len(epochs)+1, len(epochs)/20))
    plt.yticks(np.arange(0, 1.0, 0.1))
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('./result_plot/' + str(len(epochs)) + '.png')
    plt.show()

def main():
    # 固定のpickleファイルパスを指定
    history_file_path = './history/normal/20_0.pickle'
    history = load_history(history_file_path)

    # val_accをプロット
    plot_val_accuracy(history)

if __name__ == "__main__":
    main()
