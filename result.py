import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration",  type=int, default=1)
    parser.add_argument("--epochs",     type=int, default=250)
    parser.add_argument("--data_type",  type=str, default="cifar10",          choices=["stl10", "cifar100", "cifar10"])
    parser.add_argument("--model_type", type=str, default="wide_resnet_28_10", choices=["resnet18", "wide_resnet_28_10"])
    args = parser.parse_args()

    iteration  = args.iteration
    epochs     = args.epochs
    data_type  = args.data_type
    model_type = args.model_type
    
    augmentations = [
        "Original",
        # "Default",
        # "Ent-Mixup",
        "SK-Mixup",
        # "Mixup(alpha=0.5)",
        "Mixup",
        # "Mixup(alpha=2.0)",
        # "Mixup(alpha=5.0)",
        # "Manifold-Mixup(alpha=0.5)",
        # "Manifold-Mixup(alpha=1.0)",
        # "Manifold-Mixup",
        # "Manifold-Mixup(alpha=5.0)",
        # "Mixup-Curriculum",

        # "FOMA",
        # "FOMA_latent_random",

        # "FOMA_fixed_input",
        # "FOMA_fixed_latent",
        
        # "FOMA_default",
        # "FOMA_Mixup",
        
        # "FOMA_knn_input",
        # "FOMA_knn_latent",
        
        # "FOMA_hard",
        # "FOMA_curriculum"
        # "FOMA_samebatch"

        # "Mixup-Original",
        # "Mixup-PCA",
        # "Mixup-Original&PCA",
        # "PCA",
    ]
    
    ### Plot accuracy & loss ###
    plot_comparison_graph(model_type, augmentations, data_type, epochs, iteration)
    plot_comparison_graph_train(model_type, augmentations, data_type, epochs, iteration)
    
    ### Print experiments result ###
    print("========================================================================================")
    for augment in augmentations:
        print(augment)
        pickle_file_path = f"history/{model_type}/{augment}/{data_type}_{epochs}"
        avg_acc  = load_acc(pickle_file_path, iteration)
        best_acc = load_best_acc(pickle_file_path, iteration)
        avg_loss = load_loss(pickle_file_path, iteration)
        std_acc  = load_std(pickle_file_path, iteration)

        print(f"| Average Accuracy: {avg_acc}% | Best Accuracy: {best_acc}% | Test Loss: {avg_loss} | std: {std_acc} |")
        print("========================================================================================")
        
def plot_comparison_graph_train(model_type, augmentations, data_type, epoch, iteration):
    os.makedirs(f"./result_plot/{model_type}/", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    ### ACCURACY ###
    plt.subplot(1, 2, 1)
    for augment in augmentations:
        dic = {}
        for i in range(iteration):
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_{epoch}_{i}.pickle'
            with open(pickle_file_path, 'rb') as f:
                dic[i] = pickle.load(f)
                
        train_acc = np.zeros(len(dic[i]['accuracy']))
        for i in range(iteration):
            train_acc += np.array(dic[i]['accuracy'])
        train_acc = train_acc / iteration
                
        epochs = range(1, len(train_acc) + 1)

        if augment == "FOMA":
            augment = "FOMA_input"
        elif augment == "FOMA_latent_random":
            augment = "FOMA_latent"
        elif augment == "Mixup":
            augment = "Mixup(alpha=1.0)"
        elif augment == "Manifold-Mixup":
            augment = "Manifold-Mixup(alpha=2.0)"
        plt.plot(epochs, train_acc, linestyle='solid', linewidth=0.8, label=f'{augment}')
        
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0.2)
    
    ### LOSS ###
    plt.subplot(1, 2, 2)
    for augment in augmentations:
        dic = {}
        for i in range(iteration):
            pickle_file_path = f'./history/{model_type}/{augment}/{data_type}_{epoch}_{i}.pickle'
            with open(pickle_file_path, 'rb') as f:
                dic[i] = pickle.load(f)
        
        train_loss = np.zeros(len(dic[i]['loss']))
        for i in range(iteration):
            train_loss += np.array(dic[i]['loss'])
        train_loss = train_loss / iteration
        
        epochs = range(1, len(train_loss) + 1)

        if augment == "FOMA":
            augment = "FOMA_input"
        elif augment == "FOMA_latent_random":
            augment = "FOMA_latent"
        elif augment == "Mixup":
            augment = "Mixup(alpha=1.0)"
        elif augment == "Manifold-Mixup":
            augment = "Manifold-Mixup(alpha=2.0)"
        plt.plot(epochs, train_loss, linestyle='solid', linewidth=1, label=f'{augment}')
        
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./result_plot/{model_type}/{data_type}_{epoch}_train.png')
    print("Save Result!")

def plot_comparison_graph(model_type, augmentations, data_type, epoch, iteration):
    os.makedirs(f"./result_plot/{model_type}/", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    ### ACCURACY ###
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

        if augment == "FOMA":
            augment = "FOMA_input"
        elif augment == "FOMA_latent_random":
            augment = "FOMA_latent"
        elif augment == "Mixup":
            augment = "Mixup(alpha=1.0)"
        elif augment == "Manifold-Mixup":
            augment = "Manifold-Mixup(alpha=2.0)"
        plt.plot(epochs, val_acc, linestyle='solid', linewidth=0.8, label=f'{augment}')
        
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0.2)
    
    ### LOSS ###
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

        if augment == "FOMA":
            augment = "FOMA_input"
        elif augment == "FOMA_latent_random":
            augment = "FOMA_latent"
        elif augment == "Mixup":
            augment = "Mixup(alpha=1.0)"
        elif augment == "Manifold-Mixup":
            augment = "Manifold-Mixup(alpha=2.0)"
        plt.plot(epochs, val_loss, linestyle='solid', linewidth=1, label=f'{augment}')
        
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./result_plot/{model_type}/{data_type}_{epoch}.png')
    print("Save Result!")

def load_acc(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    avg_acc = 0
    for i in range(iteration):
        avg_acc += dic[i]["acc"]
    avg_acc = avg_acc / iteration
    avg_acc *= 100
    avg_acc = round(avg_acc, 2)
    return avg_acc

def load_best_acc(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    best_acc = 0
    for i in range(iteration):
        if dic[i]["acc"] >= best_acc:
            best_acc = dic[i]["acc"]
    best_acc *= 100
    best_acc = round(best_acc, 2)
    return best_acc

def load_loss(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    avg_loss = 0
    for i in range(iteration):
        avg_loss += dic[i]["loss"]
    avg_loss = avg_loss / iteration
    avg_loss = round(avg_loss, 3)
    return avg_loss

def load_std(path, iteration):
    if iteration == 0:
        return 0
    dic = {}
    for i in range(iteration):
        with open(path + "_" + str(i) + "_test.pickle", mode="rb") as f:
            dic[i] = pickle.load(f)
    acc_list = []
    for i in range(iteration):
        acc_list.append(dic[i]["acc"])
    std_acc = np.std(acc_list)
    std_acc = round(std_acc, 5)
    return std_acc

if __name__ == "__main__":
    main()
