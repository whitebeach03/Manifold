import pickle
import matplotlib.pyplot as plt

def plot_training_history(model_type, augment):
    pickle_file_path = f'./history/{model_type}/{augment}/stl10_200.pickle'
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
    plt.savefig(f'./result_plot/{model_type}/{augment}/stl10_200.png')
    # plt.show()

def load_test_history(model_type, augment):
    pickle_file_path = f'./history/{model_type}/{augment}/stl10_200_test.pickle'
    with open(pickle_file_path, 'rb') as f:
        history = pickle.load(f)
    loss = history['loss']
    acc  = history['acc'] * 100
    print(f'model_type: {model_type}, augment: {augment} -> Loss: {loss:.2f}, Acc: {acc:.2f}')


if __name__ == "__main__":

    model_type = 'resnet18'

    plot_training_history(model_type, 'normal')
    plot_training_history(model_type, 'mixup')

    load_test_history(model_type, 'normal')
    load_test_history(model_type, 'mixup')