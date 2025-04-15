import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.resnet import ResNet18
from sklearn.manifold import TSNE
from tqdm import tqdm

features_list = []
labels_list = []
data_type = "cifar10"
training = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
N = [1000, 5000, 10000]


if data_type == "cifar10":
    # checkpoint_path = f"./logs/resnet18/Original/cifar10_200_{N_train}.pth"
    checkpoint_path = f"./logs/resnet18/Manifold-Mixup/cifar10_200.pth"
    batch_size      = 128
    transform       = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset   = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    dataloader      = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
elif data_type == "stl10":
    checkpoint_path = "./logs/resnet18/Original/stl10_150.pth"
    batch_size      = 64
    transform       = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    train_dataset   = torchvision.datasets.STL10(root='./data', split='test', transform=transform, download=True)
    dataloader      = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

if training:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

model.eval()
with torch.no_grad():
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        features = model.extract_features(images)
        features_list.append(features.cpu())
        labels_list.append(labels)
X = torch.cat(features_list, dim=0).numpy()
y = torch.cat(labels_list, dim=0).numpy()

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, label="Class label")
plt.title("t-SNE of ResNet Feature Representations")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig(f"./tsne/{data_type}_ManifoldMixup")
# plt.savefig(f"./tsne/{data_type}_{N_train}")