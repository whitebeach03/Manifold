import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison_graph(model_type, augmentations, data_type, epoch, iteration):
    os.makedirs(f"./result_plot/{model_type}/", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Accuracyグラフ
    plt.subplot(1, 2, 1)
    for augment in augmentations:
        dic = {}
        for i in range(iteration):
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_{epoch}_{i}.pickle'
            with open(pickle_file_path, 'rb') as f:
                dic[i] = pickle.load(f)
                
        val_acc = np.zeros(len(dic[i]['val_accuracy']))
        for i in range(iteration):
            val_acc += np.array(dic[i]['val_accuracy'])
        val_acc = val_acc / iteration
                
        epochs = range(1, len(val_acc) + 1)
        plt.plot(epochs, val_acc, linestyle='solid', linewidth=1, label=f'{augment}')
        
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Lossグラフ
    plt.subplot(1, 2, 2)
    for augment in augmentations:
        dic = {}
        for i in range(iteration):
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_{epoch}_{i}.pickle'
            with open(pickle_file_path, 'rb') as f:
                dic[i] = pickle.load(f)
        
        val_loss = np.zeros(len(dic[i]['val_loss']))
        for i in range(iteration):
            val_loss += np.array(dic[i]['val_loss'])
        val_loss = val_loss / iteration
        
        epochs = range(1, len(val_loss) + 1)
        plt.plot(epochs, val_loss, linestyle='solid', linewidth=1, label=f'{augment}')
        
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./result_plot/{model_type}/{data_type}_{epoch}.png')

if __name__ == "__main__":
    iteration     = 1
    data_type     = "cifar10"
    epochs        = 250
    model_type    = "wide_resnet_28_10"
    augmentations = ["Original", "Mixup", "Mixup-Original", "Mixup-PCA"]
    
    plot_comparison_graph(model_type, augmentations, data_type, epochs, iteration)
