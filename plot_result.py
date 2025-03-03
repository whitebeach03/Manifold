# import pickle
# import matplotlib.pyplot as plt

# def plot_training_history(model_type, augment):
#     pickle_file_path = f'./history/{model_type}/{augment}/stl10_200.pickle'
#     with open(pickle_file_path, 'rb') as f:
#         history = pickle.load(f)

#     # エポック数を取得
#     epochs = range(1, len(history['loss']) + 1)

#     # プロットを作成
#     plt.figure(figsize=(12, 5))

#     # Lossのプロット
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history['loss'], label='Train Loss')
#     plt.plot(epochs, history['val_loss'], label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Accuracyのプロット
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history['accuracy'], label='Train Accuracy')
#     plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(f'./result_plot/{model_type}/{augment}/stl10_200.png')
#     # plt.show()

# def load_test_history(model_type, augment):
#     pickle_file_path = f'./history/{model_type}/{augment}/stl10_200_test.pickle'
#     with open(pickle_file_path, 'rb') as f:
#         history = pickle.load(f)
#     loss = history['loss']
#     acc  = history['acc'] * 100
#     print(f'model_type: {model_type}, augment: {augment} -> Loss: {loss:.2f}, Acc: {acc:.2f}')


# if __name__ == "__main__":

#     model_type = 'resnet18'

#     plot_training_history(model_type, 'normal')
#     plot_training_history(model_type, 'mixup')
#     plot_training_history(model_type, 'mixup_hidden')
#     plot_training_history(model_type, 'ours')

#     load_test_history(model_type, 'normal')
#     load_test_history(model_type, 'mixup')
#     load_test_history(model_type, 'mixup_hidden')
#     load_test_history(model_type, 'ours')


import pickle
import matplotlib.pyplot as plt

def plot_comparison_graph(model_type, augmentations):
    plt.figure(figsize=(12, 5))
    
    # Accuracyグラフ
    plt.subplot(1, 2, 1)
    for augment in augmentations:
        pickle_file_path = f'./history/{model_type}/{augment}/stl10_200.pickle'
        with open(pickle_file_path, 'rb') as f:
            history = pickle.load(f)
        epochs = range(1, len(history['accuracy']) + 1)
        # plt.plot(epochs, history['accuracy'], label=f'Train Accuracy ({augment})')
        plt.plot(epochs, history['val_accuracy'], linestyle='solid', linewidth=1, label=f'Validation Accuracy ({augment})')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Lossグラフ
    plt.subplot(1, 2, 2)
    for augment in augmentations:
        pickle_file_path = f'./history/{model_type}/{augment}/stl10_200.pickle'
        with open(pickle_file_path, 'rb') as f:
            history = pickle.load(f)
        epochs = range(1, len(history['loss']) + 1)
        # plt.plot(epochs, history['loss'], label=f'Train Loss ({augment})')
        plt.plot(epochs, history['val_loss'], linestyle='solid', linewidth=1, label=f'Validation Loss ({augment})')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./result_plot/{model_type}/comparison_augmentations.png')
    plt.show()

if __name__ == "__main__":
    model_type = 'resnet18'
    augmentations = ["Original", "Flipping", "Cropping", "Rotation", "Translation", "Noisy", "Blurring", "Random-Erasing"]
    plot_comparison_graph(model_type, augmentations)
