import pickle
import matplotlib.pyplot as plt

def plot_comparison_graph(model_type, augmentations):
    plt.figure(figsize=(12, 5))
    
    # Accuracyグラフ
    plt.subplot(1, 2, 1)
    for augment in augmentations:
        pickle_file_path = f'./history/{model_type}/{augment}/cifar10_200.pickle'
        # pickle_file_path = f"./history/{model_type}/Fine-Tuning/{augment}/cifar10_100.pickle"
        with open(pickle_file_path, 'rb') as f:
            history = pickle.load(f)
        epochs = range(1, len(history['accuracy']) + 1)
        if augment == "Manifold-Mixup-Origin":
            plt.plot(epochs, history['val_accuracy'], linestyle='solid', linewidth=1, label="Manifold-Mixup")
        elif augment == "Manifold-Mixup":
            plt.plot(epochs, history['val_accuracy'], linestyle='solid', linewidth=1, label="Manifold-Mixup(Curriculum)")
        else:
            plt.plot(epochs, history['val_accuracy'], linestyle='solid', linewidth=1, label=f'{augment}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Lossグラフ
    plt.subplot(1, 2, 2)
    for augment in augmentations:
        pickle_file_path = f'./history/{model_type}/{augment}/cifar10_200.pickle'
        # pickle_file_path = f"./history/{model_type}/Fine-Tuning/{augment}/cifar10_100.pickle"
        with open(pickle_file_path, 'rb') as f:
            history = pickle.load(f)
        epochs = range(1, len(history['loss']) + 1)
        if augment == "Manifold-Mixup-Origin":
            plt.plot(epochs, history['val_loss'], linestyle='solid', linewidth=1, label="Manifold-Mixup")
        elif augment == "Manifold-Mixup":
            plt.plot(epochs, history['val_loss'], linestyle='solid', linewidth=1, label="Manifold-Mixup(Curriculum)")
        else:
            plt.plot(epochs, history['val_loss'], linestyle='solid', linewidth=1, label=f'{augment}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./result_plot/{model_type}/comparison_manifold.png')
    plt.show()

if __name__ == "__main__":
    model_type = 'resnet18'
    augmentations = ["Original", "Mixup", "Manifold-Mixup-Origin", "PCA"]
    plot_comparison_graph(model_type, augmentations)
