import pickle
import matplotlib.pyplot as plt

def plot_comparison_graph(model_type, augmentations, data_type, N_train):
    plt.figure(figsize=(12, 5))
    
    # Accuracyグラフ
    plt.subplot(1, 2, 1)
    for augment in augmentations:
        if N_train == 40000:
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_200.pickle'
        else:
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_200_{N_train}.pickle'
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
        if N_train == 40000:
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_200.pickle'
        else:
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_200_{N_train}.pickle'
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
    plt.savefig(f'./result_plot/{model_type}/{data_type}_{N_train}.png')
    plt.show()

if __name__ == "__main__":
    model_type = 'resnet18'
    data_type = "stl10"

    if data_type == "cifar10":
        N_train = 10000
    elif data_type == "stl10":
        N_train = 40000

    augmentations = ["Original", "Mixup", "Manifold-Mixup-Origin", "PCA", "Mixup-PCA", "Mixup-PCA-oneShot"]
    # augmentations = ["Mixup-PCA", "Mixup-PCA-alpha05", "Mixup-PCA-alpha15", "Mixup-PCA-alpha20"]
    plot_comparison_graph(model_type, augmentations, data_type, N_train)
