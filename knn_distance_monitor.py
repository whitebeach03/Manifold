import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from sklearn.neighbors import NearestNeighbors

# Replace with actual model import paths
from src.models.wide_resnet import Wide_ResNet
from src.models.resnet import ResNet18

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_type, data_type, augment, epochs, num_classes, device):
    # Initialize model
    if model_type == "resnet18":
        model = ResNet18(num_classes=num_classes).to(device)
    elif model_type == "wide_resnet_28_10":
        model = Wide_ResNet(28, 10, 0.3, num_classes).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Path to saved checkpoint
    model_save_path = f"./logs/{model_type}/{augment}/{data_type}_{epochs}_0.pth"
    state = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_train_loader(data_type, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if data_type == "stl10":
        dataset = STL10(root="./data", split="train", download=True, transform=transform)
    elif data_type == "cifar100":
        dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif data_type == "cifar10":
        dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def extract_features(model, loader, device):
    feats = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            # assumes model.extract_features returns (B, D)
            feat = model.extract_features(images)
            feats.append(feat.cpu().numpy())
    feats = np.concatenate(feats, axis=0)  # (N, D)
    return feats


def compute_avg_knn_distance(features, k=10):
    # Fit k-NN (including self) and compute distances
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features)
    dists, _ = nbrs.kneighbors(features)
    # ignore self-distance (first column)
    knn_dists = dists[:, 1:]  # (N, k)
    return knn_dists.mean(axis=1).mean()


def main():
    # Parameters
    # data_type   = "cifar100"       # or "cifar100", "stl10"
    # model_type  = "wide_resnet_28_10"
    # augment     = "Default"       # folder name used for checkpoints
    # epochs_list = [5, 400]

    # # Determine dataset-specific settings
    # if data_type == "stl10":
    #     num_classes = 10; batch_size = 64
    # elif data_type == "cifar100":
    #     num_classes = 100; batch_size = 128
    # elif data_type == "cifar10":
    #     num_classes = 10; batch_size = 128
    # else:
    #     raise ValueError(f"Unknown data_type: {data_type}")

    # # Load training data
    # train_loader = get_train_loader(data_type, batch_size)

    # for epochs in epochs_list:
    #     # Load model at specified epoch
    #     model = load_model(model_type, data_type, augment, epochs, num_classes, device)
    #     # Extract features for entire training set
    #     features = extract_features(model, train_loader, device)
    #     # Compute average k-NN distance
    #     avg_dist = compute_avg_knn_distance(features, k=10)
    #     print(f"Epochs={epochs}: Avg 10-NN distance = {avg_dist:.6f}")
    
    filename = "./distance_log/wide_resnet_28_10/cifar100_400_2_knn_dist.pkl"

    with open(filename, "rb") as f:
        distance_log = pickle.load(f)
    
    epochs_list, avg_dists = zip(*distance_log)
    plt.figure(figsize=(6,4))
    plt.plot(epochs_list, avg_dists)
    plt.xlabel("Epoch")
    plt.ylabel("Average 10-NN Distance")
    plt.title(f"Local-FOMA  k-NN Distance over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result_plot/knn_distance.png")


if __name__ == '__main__':
    main()
